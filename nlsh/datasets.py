from torch.utils.data import Dataset


class RandomPositive(Dataset):

    def __init__(self, candidate_vectors, distance_func):
        self._candidate_vectors = candidate_vectors
        self._candidate_vectors_gpu = candidate_vectors.cuda()
        self._distance_func = distance_func
        self.n_candidates = self._candidate_vectors.shape[0]
        self.n_dims = self._candidate_vectors.shape[1]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        v1_idx = random.randint(0, self.n_candidates-1)
        v2_idx = random.randint(0, self.n_candidates-1)
        anchor = self._candidate_vectors[idx, :]
        anchor_gpu = self._candidate_vectors_gpu[idx, :]
        v1 = self._candidate_vectors[v1_idx, :]
        v1_gpu = self._candidate_vectors_gpu[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        v2_gpu = self._candidate_vectors_gpu[v2_idx, :]
        d1 = self._distance_func(anchor, v1, dim=0)
        d2 = self._distance_func(anchor, v2, dim=0)
        if d1 > d2:
            return anchor_gpu, v2_gpu, v1_gpu
        return anchor_gpu, v1_gpu, v2_gpu

class KNearestNeighborPositive(Dataset):
    pass
