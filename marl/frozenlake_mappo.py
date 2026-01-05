import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import gymnasium as gym


# ==========================
#  Multi-agent FrozenLake
# ==========================

class MultiAgentFrozenLake:
    """
    Несколько независимых копий FrozenLake.
    Каждый env = "агент".
    Кооперативный сценарий: общий rollout, общий централизованный критик.
    """

    def __init__(self, n_agents=4, map_name="4x4", is_slippery=False):
        self.n_agents = n_agents
        self.envs = [
            gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
            for _ in range(n_agents)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        obs = []
        for env in self.envs:
            o, _ = env.reset()
            obs.append(o)
        return np.array(obs, dtype=np.int64)

    def step(self, actions):
        """
        actions: array/list длиной n_agents
        Возвращает:
            obs: np.array [n_agents]
            rewards: np.array [n_agents]
            dones: np.array [n_agents]
        """
        next_obs, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            o, r, terminated, truncated, info = env.step(a)
            d = terminated or truncated
            next_obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.array(next_obs, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()


# ==========================
#  Вспомогательная ф-ия
# ==========================

def one_hot(indices: np.ndarray, depth: int) -> torch.Tensor:
    """
    indices: np.array shape [n_agents], значения [0..depth-1]
    """
    eye = torch.eye(depth)
    return eye[torch.from_numpy(indices)]  # [n_agents, depth]


# ==========================
#  MAPPO: Actor & Critic
# ==========================

class Actor(nn.Module):
    """
    Политика π(a|o) для каждого агента (децентрализованная).
    На вход: локальное наблюдение агента (one-hot состояния).
    На выход: распределение Categorical по действиям.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        # obs: [batch, obs_dim] или [n_agents, obs_dim]
        logits = self.net(obs)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """
    Централизованный критик V(s_global).
    На вход: конкатенация всех наблюдений агентов (глобальное состояние).
    На выход: скалярное значение V(s).
    """

    def __init__(self, global_obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        # global_obs: [batch, global_obs_dim]
        return self.net(global_obs).squeeze(-1)  # [batch]


# ==========================
#  Rollout-буфер для MAPPO
# ==========================

class MAPPORolloutBuffer:
    """
    On-policy буфер на один эпизод (до max_steps_per_episode шагов).
    Хранит:
      - локальные наблюдения агентов
      - глобальные состояния
      - действия, log π_old
      - награды и флаги завершения
      - значения V(s) от критика
    """

    def __init__(self, buffer_size: int, n_agents: int, obs_dim: int,
                 global_obs_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.device = device

        self.obs = torch.zeros(buffer_size, n_agents, obs_dim, device=device)
        self.global_obs = torch.zeros(buffer_size, global_obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, n_agents, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(buffer_size, n_agents, device=device)
        self.rewards = torch.zeros(buffer_size, n_agents, device=device)
        # scalar done per step (0/1) для эпизода в целом (все агенты завершили)
        self.dones = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)

        self.ptr = 0

    def add(self,
            obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            logprobs: torch.Tensor,
            rewards: torch.Tensor,
            dones_agents: torch.Tensor,
            value: torch.Tensor):
        """
        obs:           [n_agents, obs_dim]
        global_obs:    [global_obs_dim]
        actions:       [n_agents]
        logprobs:      [n_agents]
        rewards:       [n_agents]
        dones_agents:  [n_agents] (0/1)
        value:         scalar tensor V(s_global)
        """
        if self.ptr >= self.buffer_size:
            return  # на всякий случай, чтобы не выйти за границу

        self.obs[self.ptr] = obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = actions
        self.logprobs[self.ptr] = logprobs
        self.rewards[self.ptr] = rewards

        # глобальный done = все агенты завершили эпизод
        done_global = float(dones_agents.all().item() if dones_agents.numel() > 0 else 0.0)
        self.dones[self.ptr] = done_global
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_returns_advantages(self, gamma: float, lam: float):
        """
        GAE(γ, λ) для централизованного критика.
        Возвращает:
          advantages: [T]
          returns:    [T]
        """
        T = self.ptr
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)

        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            reward_t = self.rewards[t].mean()  # общая "кооперативная" награда
            if t == T - 1:
                next_value = 0.0
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t + 1]

            delta = reward_t + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]

        return advantages, returns


# ==========================
#  Основной тренировочный цикл MAPPO
# ==========================

def train_mappo_frozenlake_multi(
    n_agents: int = 4,
    total_episodes: int = 2000,
    max_steps_per_episode: int = 100,
    gamma: float = 0.99,
    lam: float = 0.95,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    clip_eps: float = 0.2,
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
    log_dir: str = "runs/frozenlake_mappo_multi",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MultiAgentFrozenLake(n_agents=n_agents, map_name="4x4", is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    obs_dim = n_states
    global_obs_dim = obs_dim * n_agents

    actor = Actor(obs_dim, n_actions).to(device)
    critic = Critic(global_obs_dim).to(device)

    optim_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optim_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    writer = SummaryWriter(log_dir=log_dir)
    global_step = 0

    buffer = MAPPORolloutBuffer(
        buffer_size=max_steps_per_episode,
        n_agents=n_agents,
        obs_dim=obs_dim,
        global_obs_dim=global_obs_dim,
        device=device,
    )

    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        episode_rewards_per_agent = np.zeros(n_agents, dtype=np.float32)
        buffer.ptr = 0

        # ---------- СБОР РОЛЛАУТА НА ОДИН ЭПИЗОД ----------
        for step in range(max_steps_per_episode):
            global_step += 1

            obs_oh = one_hot(obs, n_states).to(device)  # [n_agents, obs_dim]
            global_obs = obs_oh.view(-1)                # [global_obs_dim]

            dist = actor(obs_oh)                        # Categorical для каждого агента
            actions = dist.sample()                     # [n_agents]
            logprobs = dist.log_prob(actions)           # [n_agents]

            with torch.no_grad():
                value = critic(global_obs.unsqueeze(0))[0]  # scalar tensor

            actions_np = actions.cpu().numpy()
            next_obs, rewards, dones, _ = env.step(actions_np)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            dones_t = torch.tensor(dones.astype(np.float32), dtype=torch.float32, device=device)

            episode_rewards_per_agent += rewards

            buffer.add(
                obs=obs_oh,
                global_obs=global_obs,
                actions=actions,
                logprobs=logprobs,
                rewards=rewards_t,
                dones_agents=dones_t,
                value=value,
            )

            obs = next_obs

            if dones.all():
                break

        # если в эпизоде нет шагов, пропускаем
        T = buffer.ptr
        if T == 0:
            continue

        # ---------- ВЫЧИСЛЕНИЕ ADV/RET (GAE) ----------
        advantages, returns = buffer.compute_returns_advantages(gamma, lam)  # [T], [T]

        # один advantage/return на глобальное состояние → расширяем на всех агентов
        adv_all = advantages.unsqueeze(1).expand(T, n_agents)   # [T, n_agents]
        ret_all = returns.unsqueeze(1).expand(T, n_agents)      # [T, n_agents]

        # выравниваем по batch размеру: [T * n_agents, ...]
        obs_flat = buffer.obs[:T].reshape(T * n_agents, obs_dim)
        global_obs_flat = (
            buffer.global_obs[:T]
            .unsqueeze(1)
            .expand(T, n_agents, global_obs_dim)
            .reshape(T * n_agents, global_obs_dim)
        )
        actions_flat = buffer.actions[:T].reshape(T * n_agents)
        old_logprobs_flat = buffer.logprobs[:T].reshape(T * n_agents)
        adv_flat = adv_all.reshape(T * n_agents)
        ret_flat = ret_all.reshape(T * n_agents)

        # нормируем advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        dataset_size = obs_flat.size(0)
        indices = torch.randperm(dataset_size, device=device)

        # ---------- PPO ОБНОВЛЕНИЕ ----------
        for epoch in range(ppo_epochs):
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_obs = obs_flat[batch_idx]               # [B, obs_dim]
                batch_global_obs = global_obs_flat[batch_idx] # [B, global_obs_dim]
                batch_actions = actions_flat[batch_idx]       # [B]
                batch_old_logprobs = old_logprobs_flat[batch_idx]  # [B]
                batch_adv = adv_flat[batch_idx]               # [B]
                batch_returns = ret_flat[batch_idx]           # [B]

                dist = actor(batch_obs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - batch_old_logprobs).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                values = critic(batch_global_obs)
                value_loss = nn.MSELoss()(values, batch_returns)

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optim_actor.zero_grad()
                optim_critic.zero_grad()
                loss.backward()
                optim_actor.step()
                optim_critic.step()

        mean_reward = episode_rewards_per_agent.mean()
        writer.add_scalar("episode/mean_reward", mean_reward, episode)
        writer.add_scalar("loss/policy", policy_loss.item(), episode)
        writer.add_scalar("loss/value", value_loss.item(), episode)
        writer.add_scalar("stats/adv_mean", advantages.mean().item(), episode)

        if episode % 100 == 0:
            print(f"[MAPPO FrozenLake] episode={episode}/{total_episodes}, mean_r={mean_reward:.3f}")

    env.close()
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    train_mappo_frozenlake_multi()
