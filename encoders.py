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


class MultiLayerRelu(nn.Module):

    def __init__(self, input_dim, hidden_dims: List[int]):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dims = hidden_dims

        self.output_dims = hidden_dims[-1]

        self.fcs = []
        prev_dim = input_dim
        for dim in hidden_dims:
            self.fcs.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

    def forward(self, x):
        output = x
        for fc in self.fcs:
            output = F.relu(fc(output))
        return output
