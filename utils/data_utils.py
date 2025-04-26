# utils/data_utils.py
import numpy as np
import pickle
import os
import logging
from stable_baselines3.common.vec_env import DummyVecEnv

def collect_trajectories(agents_with_envs, num_episodes=200, save_dir='data/trajectories/'):
    logger = logging.getLogger(__name__)
    os.makedirs(save_dir, exist_ok=True)
    all_trajectory_sets = []

    for agent, env, metadata in agents_with_envs:
        if not isinstance(env, DummyVecEnv):
            env = DummyVecEnv([lambda: env])
        
        save_path = f"{save_dir}{metadata['env']}_{metadata['type']}_full.pkl"
        trajectories = []

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                trajectories = pickle.load(f)
            logger.info(f"Loaded trajectories from {save_path}")

        remaining_episodes = num_episodes - len(trajectories)
        if remaining_episodes <= 0:
            all_trajectory_sets.append(trajectories)
            continue

        logger.info(f"Collecting {remaining_episodes} new trajectories for {metadata['type']} in {metadata['env']}_full")
        metadata['shifted'] = False  # Always false since no shifting

        for _ in range(remaining_episodes):
            episode_trajectory = []
            obs = env.reset()
            done = False
            while not done:
                action = agent.act(obs[0])
                next_obs, reward, done, info = env.step([action])
                episode_trajectory.append({
                    'obs': obs[0].tolist(),
                    'action': float(action),
                    'reward': float(reward[0]),
                    'next_obs': next_obs[0].tolist(),
                    'done': bool(done[0]),
                    'info': info[0]
                })
                obs = next_obs
                done = done[0]
            
            # Ensure past and current are non-empty
            if len(episode_trajectory) < 2:
                logger.warning(f"Episode too short ({len(episode_trajectory)} steps), skipping")
                continue
            split_idx = max(1, int(0.8 * len(episode_trajectory)))  # At least 1 step in past
            if split_idx >= len(episode_trajectory):  # Ensure current has at least 1 step
                split_idx = len(episode_trajectory) - 1
            
            trajectories.append({
                'past': episode_trajectory[:split_idx],
                'current': episode_trajectory[split_idx:],
                'metadata': metadata
            })

        with open(save_path, 'wb') as f:
            pickle.dump(trajectories, f)
        logger.info(f"Saved {len(trajectories)} trajectories to {save_path}")
        all_trajectory_sets.append(trajectories)

    return all_trajectory_sets