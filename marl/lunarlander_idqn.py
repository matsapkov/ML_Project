import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class MultiAgentLunarLander:
    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.envs = [gym.make("LunarLander-v2") for _ in range(n_agents)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs = []
        for env in self.envs:
            o, _ = env.reset()
            obs.append(o)
        return np.array(obs, dtype=np.float32)

    def step(self, actions):
        next_obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, terminated, truncated, info = env.step(a)
            d = terminated or truncated
            next_obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.array(next_obs, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()

class QNetworkLander(nn.Module):
    def __init__(self, input_dim=8, n_actions=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)

# ==========================
#  Реплей-буфер
# ==========================

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = torch.stack([b.state for b in batch])
        actions = torch.tensor([b.action for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        next_states = torch.stack([b.next_state for b in batch])
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def train_idqn_lunarlander_multi(
    n_agents=4,
    total_episodes=3000,
    max_steps_per_episode=500,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    start_learning=2000,
    target_update_freq=1000,
    replay_capacity=200_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=2500,
    log_dir="runs/lunarlander_idqn_multi",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentLunarLander(n_agents)
    input_dim = env.observation_space.shape[0]   # 8
    n_actions = env.action_space.n               # 4

    q_net = QNetworkLander(input_dim, n_actions).to(device)
    target_net = QNetworkLander(input_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0

    def epsilon_by_episode(ep):
        frac = min(1.0, ep / eps_decay_episodes)
        return eps_start + frac * (eps_end - eps_start)

    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        episode_rewards = np.zeros(n_agents)
        eps = epsilon_by_episode(episode)

        for step in range(max_steps_per_episode):
            global_step += 1

            state_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)

            actions = np.array([
                env.action_space.sample() if random.random() < eps
                else int(torch.argmax(q_values[i]).item())
                for i in range(n_agents)
            ], dtype=np.int64)

            next_obs, rewards, dones, _ = env.step(actions)
            episode_rewards += rewards

            # Push transitions
            for i in range(n_agents):
                replay.push(
                    torch.tensor(obs[i], dtype=torch.float32),
                    actions[i],
                    rewards[i],
                    torch.tensor(next_obs[i], dtype=torch.float32),
                    float(dones[i])
                )

            obs = next_obs

            if len(replay) >= start_learning:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay.sample(batch_size)

                states_b = states_b.to(device)
                actions_b = actions_b.to(device)
                rewards_b = rewards_b.to(device)
                next_states_b = next_states_b.to(device)
                dones_b = dones_b.to(device)

                q_values_batch = q_net(states_b)
                q_sa = q_values_batch.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next = target_net(next_states_b)
                    q_next_max = q_next.max(dim=1)[0]
                    target = rewards_b + gamma * (1 - dones_b) * q_next_max

                loss = nn.MSELoss()(q_sa, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("loss", loss.item(), global_step)

            if global_step % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            if dones.all():
                break

        mean_r = episode_rewards.mean()
        writer.add_scalar("episode/mean_reward", mean_r, episode)

        if episode % 50 == 0:
            print(f"[Lander] Ep {episode} | mean_r={mean_r:.2f} | eps={eps:.3f}")

    env.close()
    writer.close()
    print("LunarLander training finished.")
