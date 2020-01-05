from time import time

import torch
import numpy as np

from .base import Trainer


def contrastive_loss(
        anchor,
        other,
        label,
        distance_func,
        negative_margin=0.1,
        positive_margin=0.0,
    ):
    d = distance_func(anchor, other)
    positive_loss = label * torch.clamp(d - positive_margin, min=0)**2
    negative_loss = (1 - label) * torch.clamp(d - negative_margin, max=0)**2
    return torch.mean(positive_loss + negative_loss) / 2


class KNearestNeighborSiamese:

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

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        label_all_np = np.random.random((len(self),)) < self._positive_rate
        label_all = torch.from_numpy(label_all_np).long().cuda()

        negative_idx_all = np.random.randint(0, len(self), (len(self),))
        negative_idx_all = torch.from_numpy(negative_idx_all).long().cuda()

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
            label_output = label_all[start:end]
            other_idx = positive_idxs * label_output + negative_idxs * (1 - label_output)
            other = self._candidate_vectors[other_idx, :]
            yield anchor, other, label_output


class KNearestNeighborLocallySiamese:

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


class SiameseTrainer(Trainer):

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
            lambda1=0.001,
            positive_margin=0.001,
            negative_margin=0.1,
            positive_rate=0.1,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger
        self._lambda1 = lambda1
        self._positive_margin = positive_margin
        self._negative_margin = negative_margin
        self._positive_rate = positive_rate

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = KNearestNeighborSiamese(
            vectors,
            self_knn,
            k=100,
            positive_rate=self._positive_rate,
        )
        return dataset

    def _get_loss(self, batch):
        anchor = self._hashing.predict(batch[0])
        other = self._hashing.predict(batch[1])
        label = batch[2]
        loss = contrastive_loss(
            anchor,
            other,
            label,
            self._hashing.distance.rowwise,
            positive_margin=self._positive_margin,
            negative_margin=self._negative_margin,
        )
        return loss
