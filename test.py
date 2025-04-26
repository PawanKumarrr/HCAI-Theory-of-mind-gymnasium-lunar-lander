import sys
import gymnasium as gym  # Import globally for consistency

def test_gymnasium():
    try:
        env = gym.make('LunarLander-v3')  # Updated to v3
        obs, info = env.reset()  # Unpack tuple from reset()
        print("gymnasium[box2d] test passed: Environment created, initial obs shape:", obs.shape)
    except ImportError as e:
        print("Failed to import gymnasium:", e)
    except Exception as e:
        print("gymnasium test failed:", e)

def test_stable_baselines3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: gym.make('LunarLander-v3')])  # Updated to v3
        model = PPO('MlpPolicy', env, verbose=0)
        print("stable-baselines3 test passed: PPO model initialized")
    except ImportError as e:
        print("Failed to import stable-baselines3:", e)
    except Exception as e:
        print("stable-baselines3 test failed:", e)

def test_torch():
    try:
        import torch
        x = torch.tensor([1.0, 2.0])
        print("torch test passed: Tensor created, sum =", x.sum().item())
    except ImportError as e:
        print("Failed to import torch:", e)
    except Exception as e:
        print("torch test failed:", e)

def test_numpy():
    try:
        import numpy as np
        arr = np.array([1, 2, 3])
        print("numpy test passed: Array created, mean =", arr.mean())
    except ImportError as e:
        print("Failed to import numpy:", e)
    except Exception as e:
        print("numpy test failed:", e)

def test_matplotlib():
    try:
        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.savefig("test_plot.png")
        plt.close()
        print("matplotlib test passed: Plot saved to test_plot.png")
    except ImportError as e:
        print("Failed to import matplotlib:", e)
    except Exception as e:
        print("matplotlib test failed:", e)

def test_imageio():
    try:
        import imageio
        import numpy as np
        frames = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(2)]  # Changed to 112x112
        imageio.mimsave("test_video.mp4", frames, fps=30)
        print("imageio test passed: Video saved to test_video.mp4")
    except ImportError as e:
        print("Failed to import imageio:", e)
    except Exception as e:
        print("imageio test failed:", e)

def main():
    print("Testing all requirements for machine_tom_framework...")
    print("Python version:", sys.version)
    print("-" * 50)
    
    test_gymnasium()
    test_stable_baselines3()
    test_torch()
    test_numpy()
    test_matplotlib()
    test_imageio()
    
    print("-" * 50)
    print("Testing complete. Check for any failure messages above.")

if __name__ == "__main__":
    main()