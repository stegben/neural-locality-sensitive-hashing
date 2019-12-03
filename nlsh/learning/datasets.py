from random import randint, random

import numpy as np
import torch
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


class OnePass(Dataset):

    def __init__(self, candidate_vectors):
        self._candidate_vectors = candidate_vectors
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        return self._candidate_vectors[idx, :]

    def batch_generator(self, batch_size, shuffle=False):
        idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idxs)

        n_batches = len(self) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            vector = self._candidate_vectors[idxs[start:end], :]
            yield vector


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
        idx = int(idx)
        anchor = self._candidate_vectors[idx, :]

        # positive sample from _candidate_self_knn
        pre_v1_idx = randint(0, self.k - 1)
        v1_idx = int(self._candidate_self_knn[idx, pre_v1_idx])

        # negative sample randomly select from the dataset
        v2_idx = randint(0, self.n_candidates - 1)

        v1 = self._candidate_vectors[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        return anchor, v1, v2

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            knn_idxs = self._candidate_self_knn[anchor_idxs[start:end], np.random.randint(0, self.k, (batch_size,))]
            positive = self._candidate_vectors[knn_idxs, :]

            random_idxs = np.random.randint(0, len(self), (batch_size,))
            negative = self._candidate_vectors[random_idxs, :]
            yield anchor, positive, negative


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

        if random() < self._positive_rate:
            # positive sample from _candidate_self_knn
            pre_v_idx = randint(0, self.k - 1)
            v_idx = self._candidate_self_knn[idx, pre_v_idx]
            label = 1
        else:
            # negative sample randomly select from the dataset
            v_idx = randint(0, self.n_candidates - 1)
            label = 0

        v = self._candidate_vectors[v_idx, :]
        return anchor, v, label


    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        label_all_np = np.random.random((len(self),)) < self._positive_rate
        label_all = torch.from_numpy(label_all_np).long().cuda()

        negative_idx_all = np.random.randint(0, len(self), (len(self),))
        # negative_idx_all = torch.from_numpy(negative_idx_all).long().cuda()

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            ramdomly_selected_positive_idx = np.random.randint(0, self.k, (batch_size,))
            positive_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_positive_idx]
            negative_idxs = negative_idx_all[start:end]
            label = label_all_np[start:end]
            label_output = label_all[start:end]
            other_idx = positive_idxs * label + negative_idxs * (1 - label)
            other = self._candidate_vectors[other_idx, :]
            yield anchor, other, label_output


class KNearestNeighborLocallySiamese(Dataset):

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            inner_k=None,
            outer_k=None,
            positive_rate=0.1,
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn
        self._positive_rate = positive_rate

        self._inner_k = inner_k or candidate_self_knn.shape[1] // 2
        self._outer_k = outer_k or candidate_self_knn.shape[1]
        if self._outer_k <= self._inner_k:
            raise ValueError(f"Outer K (got {self._outer_k}) should be larger than inner K (got {self._inner_k}).")
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        anchor = self._candidate_vectors[idx, :]

        if random() < self._positive_rate:
            # positive sample from _candidate_self_knn
            pre_v_idx = randint(0, self.k - 1)
            v_idx = self._candidate_self_knn[idx, pre_v_idx]
            label = 1
        else:
            # negative sample randomly select from the dataset
            v_idx = randint(0, self.n_candidates - 1)
            label = 0

        v = self._candidate_vectors[v_idx, :]
        return anchor, v, label


    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        label_all_np = np.random.random((len(self),)) < self._positive_rate
        label_all = torch.from_numpy(label_all_np).long().cuda()

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            ramdomly_selected_positive_idx = np.random.randint(0, self._inner_k, (batch_size,))
            ramdomly_selected_negative_idx = np.random.randint(self._inner_k, self._outer_k, (batch_size,))
            positive_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_positive_idx]
            negative_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_negative_idx]
            label = label_all_np[start:end]
            label_output = label_all[start:end]
            other_idx = positive_idxs * label + negative_idxs * (1 - label)
            other = self._candidate_vectors[other_idx, :]
            yield anchor, other, label_output
