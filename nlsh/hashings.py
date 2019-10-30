# learnable hash functions
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultivariateBernoulli:

    class _Hasher(nn.Module):

        def __init__(self, encoder, hash_size):
            super().__init__()
            self._encoder = encoder
            self.output_layer = nn.Linear(encoder.output_dim, hash_size)

        def forward(self, x):
            x = self._encoder(x)
            x = F.sigmoid(self.output_layer(x))
            return x

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

    def _binarr_to_int(self, binarr):
        out = 0
        for bit in binarr:
            out = (out << 1) | bit
        return out

    def hash(self, query_vectors):
        probs = self._hasher(query_vectors)
        codes = (probs > 0.5).tolist()
        hash_results = []
        for binarr in codes:
            hash_results.append(self._binarr_to_int(binarr))
        return hash_results

    def hash_sample(self, query_vectors, sample_size=10):
        # probs = self._hasher(query_vectors)
        # TODO: sampling hash. The larger the sample size,
        # the better recall should be
        pass


class Categorical(nn.Module):

    class _Hasher(nn.Module):

        def __init__(self, encoder, hash_size):
            super().__init__()
            self._encoder = encoder
            self.output_layer = nn.Linear(encoder.output_dim, hash_size)

        def forward(self, x):
            prob = self._encoder(x)
            prob = F.softmax(self.output_layer(prob))
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

    def hash_sample(self, query_vectors, sample_size=10):
        # probs = self._hasher(query_vectors)
        # TODO: sampling hash. The larger the sample size,
        # the better recall should be
        pass
