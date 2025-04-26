import os
import numpy as np
import torch
import random
from config import LOG_DIR, MODEL_DIR, TRAJECTORY_DIR, VIDEO_DIR, TOTAL_TIMESTEPS_LUNAR, NUM_EPISODES, ENV_LIST
from utils.logger import setup_logger
from utils.data_utils import collect_trajectories
from utils.recorder import record_video
from agents.rl_agents import train_agent, AgentWrapper
from envs.custom_envs import LunarLanderCustom

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def main():
    logger = setup_logger('run_agents', os.path.join(LOG_DIR, 'run_agents.log'))
    logger.info("Starting agent execution")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    
    reward_log_path = os.path.join(LOG_DIR, 'episode_rewards.txt')
    if os.path.exists(reward_log_path):
        os.remove(reward_log_path)
    for model_file in os.listdir(MODEL_DIR):
        if model_file.endswith('.zip'):
            os.remove(os.path.join(MODEL_DIR, model_file))
            logger.info(f"Removed existing model: {model_file}")

    agents_with_envs = []

    # LunarLander Normal (no shift, matches base env)
    env_lunar_normal = LunarLanderCustom(seed=SEED)
    total_timesteps = TOTAL_TIMESTEPS_LUNAR
    model_path = os.path.join(MODEL_DIR, 'PPO_LunarLanderCustom_full.zip')
    model = train_agent(env_lunar_normal, 'PPO', total_timesteps, model_path, LOG_DIR)
    agent_lunar_normal = AgentWrapper(model)
    agents_with_envs.append((agent_lunar_normal, env_lunar_normal, {'type': 'PPO', 'env': 'LunarLanderCustom', 'observability': 'full'}))
    record_video(agent_lunar_normal, env_lunar_normal, os.path.join(VIDEO_DIR, 'PPO_LunarLanderCustom_full.mp4'), num_episodes=10, max_steps_per_episode=1000)
    logger.info("Generated LunarLander Normal video")

    # LunarLander Shifted (optional, not in base env; comment out if not needed)
    # env_lunar_shifted = LunarLanderCustom(seed=SEED)  # No shift_enabled here either
    # model_path_shifted = os.path.join(MODEL_DIR, 'PPO_LunarLanderCustom_shifted.zip')
    # model = train_agent(env_lunar_shifted, 'PPO', total_timesteps, model_path_shifted, LOG_DIR)
    # agent_lunar_shifted = AgentWrapper(model)
    # agents_with_envs.append((agent_lunar_shifted, env_lunar_shifted, {'type': 'PPO', 'env': 'LunarLanderCustom', 'observability': 'full'}))
    # record_video(agent_lunar_shifted, env_lunar_shifted, os.path.join(VIDEO_DIR, 'shifted_LunarLanderCustom.mp4'), num_episodes=10, max_steps_per_episode=1000)
    # logger.info("Generated LunarLander Shifted video")

    trajectory_sets = collect_trajectories(agents_with_envs, NUM_EPISODES, TRAJECTORY_DIR)
    logger.info(f"Collected {len(trajectory_sets)} trajectory sets")

if __name__ == "__main__":
    main()