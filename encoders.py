from typing import List

import torch.nn as nn
import torch.nn.functional as F


class TwoLayer256Relu(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self._input_dim = input_dim
        self.output_dim = 256

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MultiLayerRelu(nn.Sequential):

    def __init__(self, input_dim, hidden_dims: List[int], with_batchnorm=False):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims

        self.output_dim = hidden_dims[-1]

        prev_dim = input_dim
        for layer_idx, dim in enumerate(hidden_dims):
            self.add_module(f"{layer_idx}_linear", nn.Linear(prev_dim, dim))
            if with_batchnorm:
                self.add_module(f"{layer_idx}_batch_norm", nn.BatchNorm1d(dim))
            self.add_module(f"{layer_idx}_relu", nn.ReLU())
            prev_dim = dim

    def forward(self, x):
        return super().forward(x)
