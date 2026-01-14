import torch
import torch.nn as nn
import torch.nn.functional as F 


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
        logits = self.gate(x) #score de confiance par expert 
        weights = F.softmax(logits, dim=-1).unsqueeze(-1)
        self.last_counts = torch.bincount(self.gate(x).argmax(dim=1),minlength=self.num_experts)

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

class MoeTopK(nn.Module):
    def __init__(self, num_experts: int, input_dim: int, hidden: int, output_dim: int, k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden, output_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts)
        )
        self.k = k
        self.num_experts = num_experts
        self.last_counts = None

    def forward(self, x):
        logits = self.gate(x)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # Create mask for top-k experts
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_k_indices, 1.0)
    
        # Combine expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Apply weights and mask
        output = torch.zeros_like(expert_outputs[:, 0, :])
        
        for _ in range(self.k):
            output += (expert_outputs * mask.unsqueeze(-1))[:, :, :].sum(dim=1)
        
        self.last_counts = torch.bincount(top_k_indices.flatten(), minlength=self.num_experts)
        
        return output / self.k
