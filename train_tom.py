# train_tom.py
from tom.tom_model import ToMnet, train_tom
from utils.logger import setup_logger
import pickle
import os
import torch
import random
import numpy as np
from config import TRAJECTORY_DIR, TOM_DIR
from torch.utils.data import Dataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

class ToMDataset(Dataset):
    def __init__(self, trajectory_sets, obs_dim, action_dim):
        self.data = []
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lunar_low = np.array([-2.5, -2.5, -10., -10., -6.2831855, -10., 0., 0.])
        self.lunar_high = np.array([2.5, 2.5, 10., 10., 6.2831855, 10., 1., 1.])
        self.lunar_range = self.lunar_high - self.lunar_low

        for traj_set in trajectory_sets:
            for idx, traj in enumerate(traj_set):
                if len(traj['past']) < 1:
                    print(f"Trajectory {idx}: Skipped due to empty past")
                    continue
                if len(traj['current']) < 2:
                    print(f"Trajectory {idx}: Skipped due to insufficient current ({len(traj['current'])})")
                    continue

                try:
                    past_data = [np.concatenate((self._normalize(np.array(t['obs'])), [float(t['action'])])) 
                                for t in traj['past']]
                    current_steps = traj['current'][:-1] if len(traj['current']) > 2 else traj['current'][:1]
                    current_data = [np.concatenate((self._normalize(np.array(t['obs'])), [float(t['action'])])) 
                                   for t in current_steps]
                    query = self._normalize(np.array(traj['current'][-1]['obs']))

                    if not past_data or not current_data:
                        print(f"Trajectory {idx}: Empty past_data or current_data after processing")
                        continue

                    past = torch.tensor(past_data, dtype=torch.float32)
                    current = torch.tensor(current_data, dtype=torch.float32)
                    query = torch.tensor(query, dtype=torch.float32)
                    target = torch.tensor(traj['current'][-1]['action'], dtype=torch.long)

                    if past.dim() != 2 or past.size(1) != obs_dim + 1 or past.size(0) == 0:
                        print(f"Trajectory {idx}: Invalid past shape {past.shape}")
                        continue
                    if current.dim() != 2 or current.size(1) != obs_dim + 1 or current.size(0) == 0:
                        print(f"Trajectory {idx}: Invalid current shape {current.shape}")
                        continue
                    if query.shape != (obs_dim,):
                        print(f"Trajectory {idx}: Invalid query shape {query.shape}")
                        continue

                    self.data.append((past, current, query, target))
                except Exception as e:
                    print(f"Error processing trajectory {idx}: {e}")
                    continue
    
    def _normalize(self, obs):
        return (obs - self.lunar_low) / self.lunar_range

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    logger = setup_logger('train_tom', 'results/logs/train_tom.log')
    trajectory_sets = []
    
    traj_file = os.path.join(TRAJECTORY_DIR, 'LunarLanderCustom_PPO_full.pkl')
    if not os.path.exists(traj_file):
        logger.error(f"Trajectory file {traj_file} not found. Run run_agents.py first.")
        raise FileNotFoundError(f"Trajectory file {traj_file} not found.")
    
    with open(traj_file, 'rb') as f:
        traj_set = pickle.load(f)
        trajectory_sets.append(traj_set)
    logger.info(f"Loaded {len(traj_set)} trajectories from {traj_file}")

    obs_dim = 8
    action_dim = 4
    dataset = ToMDataset(trajectory_sets, obs_dim, action_dim)
    logger.info(f"Filtered to {len(dataset)} valid trajectories for training")
    
    if len(dataset) == 0:
        logger.error("No valid trajectories after filtering. Check trajectory data.")
        raise ValueError("No valid trajectories available for training.")
    
    model = ToMnet(obs_dim=obs_dim, action_dim=action_dim, embed_dim=16)
    save_path = os.path.join(TOM_DIR, 'tomnet_LunarLanderCustom.pth')
    
    train_tom(
        model,
        dataset,  # Pass filtered dataset
        obs_dim=obs_dim,
        action_dim=action_dim,
        epochs=150,
        batch_size=64,
        lr=5e-4,  # Reduced from 1e-3
        save_path=save_path
    )
    logger.info("Training completed successfully.")

if __name__ == "__main__":
    main()