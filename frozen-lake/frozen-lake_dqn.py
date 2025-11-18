import gymnasium as gym
from stable_baselines3 import DQN
import tqdm

env = gym.make("FrozenLake-v1", render_mode="rgb_array")

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    exploration_fraction=0.4,
    exploration_final_eps=0.01,
    learning_starts=1000,
    buffer_size=50000,
    batch_size=32,
    target_update_interval=1000,
    learning_rate=0.001,
    gamma=0.95,
)

model.learn(total_timesteps=200_000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in tqdm.tqdm(desc="Обработка\n",iterable=range(1000)):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

    if done:
        break
