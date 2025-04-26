from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
from config import PPO_HYPERPARAMS

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0
        self.current_steps = 0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        self.current_steps += 1
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_reward)
            info = self.locals['infos'][0]
            landing = info.get('landing', False)
            env = self.locals['env'].envs[0]
            fuel_depleted = env.fuel <= 0 if hasattr(env, 'fuel') else False
            print(f"Episode {len(self.episode_rewards)}: Reward = {self.current_reward}, Steps = {self.current_steps}, Landed = {landing}, FuelOut = {fuel_depleted}")
            with open(os.path.join('results/logs', 'episode_rewards.txt'), 'a') as f:
                f.write(f"Episode {len(self.episode_rewards)}: Reward = {self.current_reward}, Steps = {self.current_steps}, Landed = {landing}, FuelOut = {fuel_depleted}\n")
            self.current_reward = 0
            self.current_steps = 0
        return True

class ScheduleCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Fraction of training completed
        fraction = self.num_timesteps / self.model._total_timesteps
        # Update learning rate and entropy coefficient
        self.model.learning_rate = 5e-4 * (1 - fraction)
        self.model.ent_coef = 0.2 * (1 - fraction)
        # Apply to optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = self.model.learning_rate
        return True

def train_agent(env, agent_type, total_timesteps=10000, model_path=None, log_dir='results/logs/'):
    if not isinstance(env, DummyVecEnv):
        env = DummyVecEnv([lambda: env])
    
    if agent_type != 'PPO':
        raise ValueError("Only PPO agent is supported.")
    
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"Loaded PPO model from {model_path}")
    else:
        # Heuristic warm-start
        def heuristic_policy(obs):
            y, vy = obs[1], obs[3]
            x = obs[0]
            if y > 0.3:  # Fire main engine when above ground
                return 2
            if x < -0.1:  # Left drift, fire right engine
                return 3
            if x > 0.5:  # Right drift, fire left engine
                return 1
            return 0
        
        obs = env.reset()
        for _ in range(50000):  # Extended to 50k steps
            action = heuristic_policy(obs[0])
            obs, reward, done, _ = env.step([action])
            if done:
                obs = env.reset()

        log_name = f"PPO_{env.envs[0].spec.id}"
        tensorboard_log_dir = os.path.join(log_dir, 'tensorboard', log_name)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_log_dir, **PPO_HYPERPARAMS)
        reward_callback = RewardCallback()
        schedule_callback = ScheduleCallback()
        model.learn(total_timesteps=total_timesteps, callback=[reward_callback, schedule_callback])
        if model_path:
            model.save(model_path)
            print(f"Trained and saved PPO model to {model_path}")
    
    return model

class AgentWrapper:
    def __init__(self, model):
        self.model = model
    
    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action.item() if isinstance(action, np.ndarray) else int(action)