#recorder.py

# utils/recorder.py
import gymnasium as gym
import imageio
import os
import logging
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

def record_video(agent, env, video_path, num_episodes=10, max_steps_per_episode=500, fps=30):
    logger = logging.getLogger(__name__)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frames = []

    # Unwrap DummyVecEnv if present
    if isinstance(env, DummyVecEnv):
        env = env.envs[0]

    for episode in range(num_episodes):
        obs, _ = env.reset()
        for step in range(max_steps_per_episode):
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            # Ensure frame is 400x600 (LunarLander-v3 default)
            if frame.shape[0] != 400 or frame.shape[1] != 600:
                frame = np.pad(frame, ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)[:400, :600, :]
            frames.append(frame)
            if terminated or truncated:
                break
        logger.info(f"Recorded episode {episode + 1}/{num_episodes}")

    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    logger.info(f"Saved video with {num_episodes} episodes to {video_path}")