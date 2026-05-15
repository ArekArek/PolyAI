import torch
import torch.nn as nn
import numpy as np
import utils
from utils import CONFIG


class ModelMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = CONFIG["training"]["hidden_size"]
        self.num_layers = CONFIG["training"]["layers_count"]
        
        self.mlp = nn.Sequential(
            nn.Linear((CONFIG["polynomial_degree"] + 1) * 2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, CONFIG["polynomial_degree"]*2)
        )


    def forward(self, x):
        flat = torch.stack([x.real, x.imag], dim=-1).flatten(-2)
        out = self.mlp(flat)
￼       
        out = out.unflatten(-1, (CONFIG["polynomial_degree"], 2))  # (B, 5, 2)
        return torch.complex(out[..., 0], out[..., 1])
