import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)  
    
class MoE(nn.Module):
    """
    MoE Soft Gate - tous les experts sont actifs avec des poids continus
    """
    def __init__(self, num_experts: int, input_dim: int, hidden: int, output_dim: int):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden, output_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim,hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,num_experts)
        )
        self.last_counts = None
        self.num_experts = num_experts
    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        
        self.last_counts = weights.squeeze(-1).mean(dim=0).cpu().detach().numpy()

        output = (expert_outputs * weights).sum(dim=1) 
        return output

class DenseModel(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)
