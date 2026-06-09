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
        
        layers = []

        layers.append(nn.Linear((CONFIG["polynomial_degree"] + 1) * 2, self.hidden_size))
        layers.append(nn.Tanh()) # normalize data, as tanh gives values -1 < x < 1

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(self.hidden_size, CONFIG["polynomial_degree"]*2))

        
        self.mlp = nn.Sequential(*layers)


    def forward(self, x):
        flat = x.flatten(-2) 
        
        out = self.mlp(flat)

        out = out.unflatten(-1, (CONFIG["polynomial_degree"], 2))  # (B, 5, 2)
        return out
