import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs" / "sb3"
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("Taxi-v3")

model = A2C(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(LOG_DIR),
    learning_rate=5e-5,
    gamma=0.99,
    n_steps=128,
    ent_coef=0.01,
)

model.learn(total_timesteps=8_000_000, tb_log_name="A2C_Taxi")

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

print(f"Mean reward over {num_episodes} episodes (A2C Taxi): {np.mean(all_rewards):.3f}")
