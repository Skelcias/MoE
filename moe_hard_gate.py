
from math import exp
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

class MoETopK(nn.Module):
    """
    MoE avec TopK Hard Gating
    """
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
        self.output_dim = output_dim

    def forward(self, x):
        B = x.size(0) 
        gate_logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1) 

        output = torch.zeros(B,self.output_dim,device=x.device)
        for e, expert in enumerate(self.experts):
            mask = (top_k_indices == e)
            if not mask.any():
                continue

            batch_mask = mask.any(dim=1)
            x_sub = x[batch_mask]
            y_sub = expert(x_sub)
            w = top_k_weights[mask]
            w = w.unsqueeze(-1)

            output[batch_mask] += y_sub * w 
        self.last_counts = torch.bincount(top_k_indices.flatten(),minlength=self.num_experts)
        return output    


