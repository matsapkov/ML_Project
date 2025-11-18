import gymnasium as gym
from stable_baselines3 import PPO
import torch

# Создание среды
env = gym.make("FrozenLake-v1", render_mode="rgb_array")
print(torch.cuda.is_available())  # Вернет True, если CUDA доступна
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=32,
    learning_rate=0.001,
    gamma=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cuda",
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model is using device: {device}")

model.learn(total_timesteps=200_000)

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

    if done:
        break
