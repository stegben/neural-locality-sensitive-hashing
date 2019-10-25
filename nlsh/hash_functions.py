# learnable hash functions
import torch.nn as nn
import torch.nn.functional as F


class MultivariateBernoulli(nn.Module):

    def __init__(self, input_dim: int, hash_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, hash_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.output_layer(x))
        return x


class Categorical(nn.Module):

    def __init__(self, input_dim: int, hash_size: int):
        super().__init__()
        output_size = 2 ** hash_size
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output_layer(x))
        return x


Encoder = MultivariateBernoulli
