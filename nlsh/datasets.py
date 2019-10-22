import random
from torch.utils.data import Dataset


class _Datasets:
    pass


class RandomPositive(Dataset):

    def __init__(self, candidate_vectors, distance_func):
        self._candidate_vectors = candidate_vectors
        self._distance_func = distance_func
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        v1_idx = random.randint(0, self.n_candidates - 1)
        v2_idx = random.randint(0, self.n_candidates - 1)
        anchor = self._candidate_vectors[idx, :]
        v1 = self._candidate_vectors[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        d1 = self._distance_func(anchor, v1, dim=0)
        d2 = self._distance_func(anchor, v2, dim=0)
        if d1 > d2:
            return anchor, v2, v1
        return anchor, v1, v2
