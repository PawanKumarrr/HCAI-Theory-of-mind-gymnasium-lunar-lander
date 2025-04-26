import pickle
with open('data/trajectories/LunarLanderCustom_PPO_full.pkl', 'rb') as f:
    traj_set = pickle.load(f)
    for i, traj in enumerate(traj_set):
        if len(traj['past']) == 0:
            print(f"Trajectory {i}: Empty past")
        elif len(traj['current']) < 2:
            print(f"Trajectory {i}: Current too short: {len(traj['current'])}")