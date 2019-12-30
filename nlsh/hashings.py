# learnable hash functions
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultivariateBernoulli:

    class _Hasher(nn.Module):

        def __init__(self, encoder, hash_size, tanh_output=False):
            super().__init__()
            self._encoder = encoder
            self._tanh_output = tanh_output
            self.output_layer = nn.Linear(encoder.output_dim, hash_size)

        def forward(self, x):
            x = self._encoder(x)
            if self._tanh_output:
                x = torch.tanh(self.output_layer(x))
            else:
                x = torch.sigmoid(self.output_layer(x))
            return x

    def __init__(self, encoder, hash_size, distance_func, tanh_output=False):
        self._encoder = encoder
        self._hash_size = hash_size

        # TODO: refactor. distance_dunc should not be the member of hashings
        self._distance_func = distance_func
        self._tanh_output = tanh_output

        self._hasher = self._Hasher(self._encoder, self._hash_size, tanh_output).cuda()

    def predict(self, x):
        return self._hasher(x)

    @property
    def distance(self):
        return self._distance_func

    @property
    def output_dim(self):
        return self._hash_size

    def parameters(self):
        return self._hasher.parameters()

    def save(self, base_name):
        scripted_model_cpu = torch.jit.script(self._hasher.cpu())
        torch.jit.save(scripted_model_cpu, base_name+"_cpu.pt")
        scripted_model_gpu = torch.jit.script(self._hasher.cuda())
        torch.jit.save(scripted_model_gpu, base_name+"_gpu.pt")
    # TODO: implement a `load` classmethod

    def train_mode(self, on):
        if on:
            self._hasher.train()
        else:
            self._hasher.eval()

    def _binarr_to_int(self, binarr):
        out = 0
        for bit in binarr:
            out = (out << 1) | bit
        return out

    def hash(self, query_vectors, n=1):
        probs = self._hasher(query_vectors)
        if self._tanh_output:
            probs = probs / 2. + 0.5
        dist = torch.distributions.Bernoulli(probs)

        base_codes = (dist.probs > 0.5).unsqueeze_(1)

        if n == 1:
            # hard hash
            codes = base_codes.tolist()
        elif n > 1:
            # sample hash
            # (batch_size, n, code_size)
            sampled_codes = dist.sample((n - 1,)).int().permute(1, 0, 2)
            codes = torch.cat((base_codes.int(), sampled_codes), dim=1).tolist()
        else:
            raise ValueError(f"`n` should be positive integer, but got {n}")

        hash_results = []
        for binarrs in codes:
            hashes = []
            for binarr in binarrs:
                hashes.append(self._binarr_to_int(binarr))
            hash_results.append(set(hashes))
        return hash_results


class Categorical:

    class _Hasher(nn.Module):

        def __init__(self, encoder, hash_size):
            super().__init__()
            self._encoder = encoder
            self.output_layer = nn.Linear(encoder.output_dim, hash_size)

        def forward(self, x):
            prob = self._encoder(x)
            prob = F.softmax(self.output_layer(prob), dim=1)
            return prob

    def __init__(self, encoder, hash_size, distance_func):
        self._encoder = encoder
        self._hash_size = hash_size
        self._distance_func = distance_func

        self._hasher = self._Hasher(self._encoder, self._hash_size).cuda()

    def predict(self, x):
        return self._hasher(x)

    def distance(self, y1, y2):
        return self._distance_func(y1, y2)

    def parameters(self):
        return self._hasher.parameters()

    def save(self, base_name):
        scripted_model_cpu = torch.jit.script(self._hasher.cpu())
        torch.jit.save(scripted_model_cpu, base_name+"_cpu.pt")
        scripted_model_gpu = torch.jit.script(self._hasher.cuda())
        torch.jit.save(scripted_model_gpu, base_name+"_gpu.pt")

    def hash(self, query_vectors):
        probs = self._hasher(query_vectors)
        return probs.argmax(axis=1).tolist()

    def train_mode(self, on):
        if on:
            self._hasher.train()
        else:
            self._hasher.eval()


class ProductQuantization:

    def __init__(self, bits_of_each_band: List[int]):
        pass
