from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs" / "sb3"
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("LunarLander-v3")

model = PPO(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(LOG_DIR),
    n_steps=1024,
    batch_size=64,
    learning_rate=6.3e-4,
    gamma=0.999,
    n_epochs=4,
    ent_coef=0.01,
    gae_lambda=0.98,
)

model.learn(total_timesteps=8000000, tb_log_name="PPO_LunarLander")

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
