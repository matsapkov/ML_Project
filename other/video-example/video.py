import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os

# Папка для сохранения видео
video_folder = os.path.join(os.getcwd(), "videos")
os.makedirs(video_folder, exist_ok=True)

# Среда с визуализацией
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Запись всех эпизодов
env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
print(f"🎬 Видео сохранено в: {video_folder}")
