import gymnasium as gym
import numpy as np

class LunarLanderCustom(gym.Wrapper):
    def __init__(self, seed=None):
        # Exact arguments from documentation
        env = gym.make(
            "LunarLander-v3",
            continuous=False,         # Discrete actions as specified
            gravity=-10.0,            # Default gravity
            enable_wind=False,        # Wind off by default
            wind_power=15.0,          # Default wind power (unused unless wind enabled)
            turbulence_power=1.5,     # Default turbulence power (unused unless wind enabled)
            render_mode='rgb_array'   # For video recording, doesn’t affect dynamics
        )
        super().__init__(env)
        if seed is not None:
            self.env.reset(seed=seed)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # No custom modifications; helipad is at (0, 0) per base env
        info['landing'] = False  # Minimal addition for RewardCallback compatibility
        return obs, info  # Raw observation, no normalization

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        x = next_obs[0]
        y = next_obs[1]
        vx = next_obs[2]
        vy = next_obs[3]
        helipad_x1 = self.env.unwrapped.helipad_x1
        helipad_x2 = self.env.unwrapped.helipad_x2
        print(f"x={x}, y={y}, vy={vy}, helipad=({helipad_x1}, {helipad_x2}), legs=({next_obs[6]}, {next_obs[7]})")

        # Landing detection for logging only, no reward modification
        if (helipad_x1 <= x <= helipad_x2 and y <= 0.1) or (next_obs[6] == 1 and next_obs[7] == 1):
            info['landing'] = True
            print(f"Landing detected! x={x}, y={y}, helipad=({helipad_x1}, {helipad_x2})")

        # Base env handles all termination (crash, x > 1, not awake); no custom max_steps
        return next_obs, reward, terminated, truncated, info

    def _normalize(self, obs):
        # No normalization to match base env
        return obs