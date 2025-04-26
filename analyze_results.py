# analyze_results.py
from tom.tom_model import ToMnet
import torch
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import TRAJECTORY_DIR, TOM_DIR
from utils.logger import setup_logger

PLOT_DIR = 'results/plots/'
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_training_loss(log_file, save_path):
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Loss' in line:
                loss = float(line.split('Loss: ')[1].split()[0])
                losses.append(loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(true_actions, pred_actions, action_dim, save_path):
    cm = np.zeros((action_dim, action_dim))
    for true, pred in zip(true_actions, pred_actions):
        cm[true, pred] += 1
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Action')
    plt.ylabel('True Action')
    plt.savefig(save_path)
    plt.close()

def plot_prob_heatmap(probs, true_actions, save_path):
    probs_np = np.vstack([p.detach().numpy() for p in probs])
    plt.figure(figsize=(10, 6))
    sns.heatmap(probs_np, cmap='viridis', xticklabels=[f'Action {i}' for i in range(probs_np.shape[1])])
    plt.scatter(np.zeros(len(true_actions)), range(len(true_actions)), c=true_actions, cmap='tab10', s=50, edgecolors='k')
    plt.title('Prediction Probability Heatmap')
    plt.xlabel('Action')
    plt.ylabel('Trajectory')
    plt.savefig(save_path)
    plt.close()

def plot_action_time_series(traj, model, obs_dim, save_path):
    probs = []
    for i in range(len(traj['current'])):
        past = [list(t['obs']) + [float(t['action'])] for t in traj['past']]
        current = [list(t['obs']) + [float(t['action'])] for t in traj['current'][:i]]
        query = list(traj['current'][i]['obs'])
        
        past = torch.tensor(past, dtype=torch.float32).unsqueeze(0)
        current = torch.tensor(current, dtype=torch.float32).unsqueeze(0) if current else torch.zeros(1, 0, obs_dim + 1)
        query = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
        
        pred = model(past, current, query)
        probs.append(pred.squeeze().detach().numpy())
    
    probs = np.array(probs)
    plt.figure(figsize=(12, 6))
    for i in range(probs.shape[1]):
        plt.plot(probs[:, i], label=f'Action {i}')
    plt.title('Action Probability Time Series')
    plt.xlabel('Step in Trajectory')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    logger = setup_logger('analyze_results', 'results/logs/analyze_results.log')
    
    # Plot training loss if log exists
    log_file = 'results/logs/train_tom.log'
    if os.path.exists(log_file):
        plot_training_loss(log_file, os.path.join(PLOT_DIR, 'training_loss.png'))
    else:
        logger.warning(f"Training log {log_file} not found. Run train_tom.py first.")

    # Load trajectories
    traj_file = os.path.join(TRAJECTORY_DIR, 'LunarLanderCustom_PPO_full.pkl')
    if not os.path.exists(traj_file):
        logger.error(f"Trajectory file {traj_file} not found. Run run_agents.py first.")
        raise FileNotFoundError(f"Trajectory file {traj_file} not found.")
    
    with open(traj_file, 'rb') as f:
        traj_set = pickle.load(f)
    logger.info(f"Loaded {len(traj_set)} trajectories from {traj_file}")

    # Load ToMnet model
    obs_dim = 8
    action_dim = 4
    model_path = os.path.join(TOM_DIR, 'tomnet_LunarLanderCustom.pth')
    if not os.path.exists(model_path):
        logger.error(f"ToMnet model {model_path} not found. Run train_tom.py first.")
        raise FileNotFoundError(f"ToMnet model {model_path} not found.")
    
    model = ToMnet(obs_dim=obs_dim, action_dim=action_dim, embed_dim=16)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info(f"Loaded ToMnet model from {model_path}")

    # Analyze trajectories
    correct = 0
    total = 0
    all_true_actions = []
    all_pred_actions = []
    all_probs = []

    # Per-file analysis (first 5 trajectories for plots)
    file_true = []
    file_pred = []
    file_probs = []
    
    for i, traj in enumerate(traj_set[:5]):
        try:
            past = [list(t['obs']) + [float(t['action'])] for t in traj['past']]
            current = [list(t['obs']) + [float(t['action'])] for t in traj['current'][:-1]]
            query = list(traj['current'][-1]['obs'])
            
            past = torch.tensor(past, dtype=torch.float32).unsqueeze(0)
            current = torch.tensor(current, dtype=torch.float32).unsqueeze(0)
            query = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
            
            pred = model(past, current, query)
            true_action = int(traj['current'][-1]['action'])
            pred_action = torch.argmax(pred, dim=1).item()
            
            is_correct = pred_action == true_action
            correct += is_correct
            total += 1
            
            file_true.append(true_action)
            file_pred.append(pred_action)
            file_probs.append(pred)
            all_true_actions.append(true_action)
            all_pred_actions.append(pred_action)
            all_probs.append(pred)
            
            logger.info(f"Trajectory {i+1} - Predicted: {pred_action}, True: {true_action}, Correct: {is_correct}, Query Obs: {query[0].tolist()}")
        except Exception as e:
            logger.warning(f"Trajectory {i+1} skipped due to error: {e}")

    plot_confusion_matrix(file_true, file_pred, action_dim, os.path.join(PLOT_DIR, 'cm_LunarLanderCustom_PPO_full.png'))
    plot_prob_heatmap(file_probs, file_true, os.path.join(PLOT_DIR, 'heatmap_LunarLanderCustom_PPO_full.png'))
    plot_action_time_series(traj_set[0], model, obs_dim, os.path.join(PLOT_DIR, 'timeseries_LunarLanderCustom_PPO_full.png'))

    # Overall accuracy with false-belief simulation
    correct_normal = 0
    total_normal = 0
    correct_decoy = 0
    total_decoy = 0
    normal_true = []
    normal_pred = []
    decoy_true = []
    decoy_pred = []
    
    for i, traj in enumerate(traj_set[5:]):
        try:
            past = [list(t['obs']) + [float(t['action'])] for t in traj['past']]
            current = [list(t['obs']) + [float(t['action'])] for t in traj['current'][:-1]]
            query = list(traj['current'][-1]['obs'])
            
            # Simulate false belief: Assume PPO believes helipad is shifted right by 0.2
            is_decoy = (i % 2 == 0)  # Alternate between normal and decoy
            query_modified = query.copy()
            if is_decoy:
                # Modify x-coordinate to reflect PPO's "belief" (helipad at original position)
                query_modified[0] -= 0.2  # Shift x left to simulate original helipad position
            
            past = torch.tensor(past, dtype=torch.float32).unsqueeze(0)
            current = torch.tensor(current, dtype=torch.float32).unsqueeze(0)
            query = torch.tensor(query_modified, dtype=torch.float32).unsqueeze(0)
            
            pred = model(past, current, query)
            true_action = int(traj['current'][-1]['action'])
            pred_action = torch.argmax(pred, dim=1).item()
            
            is_correct = pred_action == true_action
            if is_decoy:
                correct_decoy += is_correct
                total_decoy += 1
                decoy_true.append(true_action)
                decoy_pred.append(pred_action)
            else:
                correct_normal += is_correct
                total_normal += 1
                normal_true.append(true_action)
                normal_pred.append(pred_action)
            
            all_true_actions.append(true_action)
            all_pred_actions.append(pred_action)
            all_probs.append(pred)
        except Exception as e:
            logger.warning(f"Trajectory skipped due to error: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    normal_accuracy = (correct_normal / total_normal) * 100 if total_normal > 0 else 0
    decoy_accuracy = (correct_decoy / total_decoy) * 100 if total_decoy > 0 else 0
    
    print(f"Overall Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"Normal Accuracy: {correct_normal}/{total_normal} ({normal_accuracy:.2f}%)")
    print(f"Decoy Accuracy: {correct_decoy}/{total_decoy} ({decoy_accuracy:.2f}%)")
    logger.info(f"Overall Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    logger.info(f"Normal Accuracy: {correct_normal}/{total_normal} ({normal_accuracy:.2f}%)")
    logger.info(f"Decoy Accuracy: {correct_decoy}/{total_decoy} ({decoy_accuracy:.2f}%)")
    
    plot_confusion_matrix(all_true_actions, all_pred_actions, action_dim, os.path.join(PLOT_DIR, 'cm_overall.png'))
    plot_prob_heatmap(all_probs, all_true_actions, os.path.join(PLOT_DIR, 'heatmap_overall.png'))

if __name__ == "__main__":
    main()