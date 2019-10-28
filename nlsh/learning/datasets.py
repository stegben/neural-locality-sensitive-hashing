import random
from torch.utils.data import Dataset


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


class KNearestNeighborTriplet(Dataset):

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            k=None,
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn

        self.k = k or candidate_self_knn.shape[1]
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        anchor = self._candidate_vectors[idx, :]

        # positive sample from _candidate_self_knn
        pre_v1_idx = random.randint(0, self.k - 1)
        v1_idx = self._candidate_self_knn[idx, pre_v1_idx]

        # negative sample randomly select from the dataset
        v2_idx = random.randint(0, self.n_candidates - 1)

        v1 = self._candidate_vectors[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        return anchor, v1, v2


class KNearestNeighborSiamese(Dataset):

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            k=None,
            positive_rate=0.1,
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn
        self._positive_rate = positive_rate

        self.k = k or candidate_self_knn.shape[1]
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        anchor = self._candidate_vectors[idx, :]

        if random.random() < self._positive_rate:
            # positive sample from _candidate_self_knn
            pre_v_idx = random.randint(0, self.k - 1)
            v_idx = self._candidate_self_knn[idx, pre_v_idx]
            label = 1
        else:
            # negative sample randomly select from the dataset
            v_idx = random.randint(0, self.n_candidates - 1)
            label = 0

        v = self._candidate_vectors[v_idx, :]
        return anchor, v, label
