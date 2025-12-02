import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs" / "sb3"
LOG_DIR.mkdir(parents=True, exist_ok=True)

env = gym.make("FrozenLake-v1", is_slippery=True)

model = A2C(
    "MlpPolicy",
    env,
    verbose=0,
    tensorboard_log=str(LOG_DIR),
    learning_rate=7e-4,
    gamma=0.99,
    n_steps=32,
    ent_coef=0.01,
)

stop_callback = StopTrainingOnRewardThreshold(
    reward_threshold=0.8,
    verbose=1
)

eval_callback = EvalCallback(
    env,
    callback_on_new_best=stop_callback,
    eval_freq=5000,
    n_eval_episodes=100,
    verbose=0
)

model.learn(total_timesteps=300_000, tb_log_name="A2C_FrozenLake", callback=eval_callback)

vec_env = model.get_env()
obs = vec_env.reset()

all_episode_rewards = []
num_episodes = 1000

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

    all_episode_rewards.append(ep_reward)
    obs = vec_env.reset()

mean_reward = np.mean(all_episode_rewards)
print(f"Mean reward over {num_episodes} episodes: {mean_reward:.3f}")
