# tom/tom_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from torch.nn.utils.rnn import pad_sequence

class ToMnet(nn.Module):
    def __init__(self, obs_dim, action_dim, embed_dim=16):
        super().__init__()
        self.character_net = nn.Sequential(
            nn.Linear(obs_dim + 1, 128),
            nn.ReLU(),
            nn.LSTM(128, embed_dim, batch_first=True)
        )
        self.mental_net = nn.Sequential(
            nn.Linear(obs_dim + 1 + embed_dim, 128),
            nn.ReLU(),
            nn.LSTM(128, embed_dim, batch_first=True)
        )
        self.prediction_net = nn.Sequential(
            nn.Linear(obs_dim + 2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, past_traj, current_traj, query_obs):
        _, (h_n, _) = self.character_net[2](self.character_net[:2](past_traj))
        echar = h_n[-1]
        
        if current_traj.size(1) > 0:
            current_input = torch.cat([current_traj, echar.unsqueeze(1).repeat(1, current_traj.size(1), 1)], dim=-1)
            _, (h_n, _) = self.mental_net[2](self.mental_net[:2](current_input))
            emental = h_n[-1]
        else:
            emental = torch.zeros_like(echar)
        
        pred_input = torch.cat([query_obs, echar, emental], dim=-1)
        return self.prediction_net(pred_input)

def custom_collate_fn(batch):
    pasts, currents, queries, targets = zip(*batch)
    for i, (p, c) in enumerate(zip(pasts, currents)):
        print(f"Batch {i}: past shape={p.shape}, current shape={c.shape}")
        if p.dim() != 2 or p.size(1) != 9 or p.size(0) == 0:
            raise ValueError(f"Invalid past shape in batch {i}: {p.shape}")
        if c.dim() != 2 or c.size(1) != 9 or c.size(0) == 0:
            raise ValueError(f"Invalid current shape in batch {i}: {c.shape}")
    pasts_padded = pad_sequence(pasts, batch_first=True, padding_value=0.0)
    currents_padded = pad_sequence(currents, batch_first=True, padding_value=0.0)
    queries = torch.stack(queries)
    targets = torch.stack(targets)
    return pasts_padded, currents_padded, queries, targets

def train_tom(model, dataset, obs_dim, action_dim, epochs=50, batch_size=64, lr=1e-3, save_path='data/tom/tomnet.pth'):
    logger = setup_logger('train_tom', 'results/logs/train_tom.log')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    # Add class weights to address imbalance
    class_counts = [8524, 455, 145, 876]  # From your cm_overall.png
    class_weights = [1.0 / count for count in class_counts]
    total = sum(class_weights)
    class_weights = [w / total for w in class_weights]  # Normalize
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for past, current, query, target in dataloader:
            optimizer.zero_grad()
            pred = model(past, current, query)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved ToMnet to {save_path}")