import torch
import torch.nn as nn
import numpy as np
import utils
from utils import CONFIG


class ModelGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 3

        # input size is 2 (real and imaginary part)
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=CONFIG["training"]["hidden_size"],
            num_layers=CONFIG["training"]["layers_count"],
            batch_first=True,
            dropout=0.1,
        )

        # map state into polynomial_degree * 2 (real and imaginary part for each zeroe)
        self.output_projection = nn.Linear(
            CONFIG["training"]["hidden_size"], CONFIG["polynomial_degree"] * 2
        )

    def forward(self, x):
        # flow throught GRU
        out, h_n = self.gru(x)

        # pick last element, which is "summary" for sequence of zeroes
        last_step = out[:, -1, :]

        # map to output format polynomial_degree * 2 (real and imaginary part for each zeroe)
        out = self.output_projection(last_step)  # (B, 10)
        out = out.unflatten(-1, (CONFIG["polynomial_degree"], 2))  # (B, 5, 2)

        return out
