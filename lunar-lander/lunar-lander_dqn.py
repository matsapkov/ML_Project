from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs" / "sb3"
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("LunarLander-v3")

model = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(LOG_DIR),
    gamma=0.99,
    n_steps=5,
    learning_rate=6.3e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    train_freq=4,
    gradient_steps=-1,
    exploration_fraction=0.12,
    exploration_final_eps=0.1,
    policy_kwargs={"net_arch": [256, 256]},
)

model.learn(total_timesteps=300000, tb_log_name="DQN_LunarLander")

vec_env = model.get_env()
obs = vec_env.reset()

all_episode_rewards = []
num_episodes = 1000
for _ in range(num_episodes):
    episode_rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        reward = reward[0]
        done = done[0]
        episode_rewards.append(reward)

    all_episode_rewards.append(sum(episode_rewards))
    obs = vec_env.reset()

print(f"Mean reward: {np.mean(all_episode_rewards):.3f}")
