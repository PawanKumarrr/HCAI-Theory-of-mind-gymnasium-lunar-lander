# config.py
LOG_DIR = 'results/logs/'
MODEL_DIR = 'data/models/'
TRAJECTORY_DIR = 'data/trajectories/'
VIDEO_DIR = 'results/videos/'
TOM_DIR = 'data/tom/'

TOTAL_TIMESTEPS_LUNAR = 8000000  # Increased to 7M as planned
NUM_EPISODES = 10000
ENV_LIST = ['LunarLanderCustom']
AGENT_LIST = ['PPO']

PPO_HYPERPARAMS = {
    'learning_rate': 5e-4,  # Initial value, will decay via callback
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 20,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'ent_coef': 0.2,  # Initial value, will decay via callback
    'vf_coef': 0.5,
    'clip_range': 0.2,
    'max_grad_norm': 0.5,
    'target_kl': None,
}