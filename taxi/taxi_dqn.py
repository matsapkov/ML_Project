import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs" / "sb3"
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("Taxi-v3")

model = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(LOG_DIR),
    learning_rate=5e-4,
    buffer_size=100_000,
    batch_size=64,
    learning_starts=10_000,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.9,
    exploration_final_eps=0.1,
)

model.learn(total_timesteps=1_000_000, tb_log_name="DQN_Taxi")

vec_env = model.get_env()
obs = vec_env.reset()

all_rewards = []
num_episodes = 500

for _ in range(num_episodes):
    ep_reward = 0.0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        r = reward[0]
        d = done[0]

        ep_reward += r
        done = d

    all_rewards.append(ep_reward)
    obs = vec_env.reset()

print(f"Mean reward over {num_episodes} episodes (DQN Taxi): {np.mean(all_rewards):.3f}")
