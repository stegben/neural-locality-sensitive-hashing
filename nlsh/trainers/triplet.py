from time import time

import torch
import numpy as np

from .base import Trainer


def triplet_loss(
        anchor,  # (n, d)
        pos,  # (n, d)
        neg,  # (n, d)
        distance_func,
        margin=0.1,
    ):
    d_pos = distance_func(anchor, pos)  # (n)
    d_neg = distance_func(anchor, neg)  # (n)
    loss_vec = torch.clamp(d_pos - d_neg + margin, min=0)
    return torch.mean(loss_vec)


class KNearestNeighborTriplet:

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

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        if shuffle:
            anchor_idxs = torch.randperm(len(self)).cuda()
        else:
            anchor_idxs = torch.arange(len(self)).cuda()
        knn_col_idxs = torch.randint(0, self.k, (len(self),))
        negative_random_idxs = torch.randint(0, len(self), (len(self),))

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            knn_idxs = self._candidate_self_knn[anchor_idxs[start:end], knn_col_idxs[start:end]]
            positive = self._candidate_vectors[knn_idxs, :]

            negative = self._candidate_vectors[negative_random_idxs[start:end], :]
            yield anchor, positive, negative

class TripletTrainer(Trainer):

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
            lambda1=0.001,
            margin=0.1,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger
        self._lambda1 = lambda1
        self._margin = margin

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = KNearestNeighborTriplet(
            vectors,
            self_knn,
            k=100,
        )
        return dataset

    def _get_loss(self, batch):
        anchor = self._hashing.predict(batch[0])
        positive = self._hashing.predict(batch[1])
        negative = self._hashing.predict(batch[2])
        loss = triplet_loss(
            anchor,
            positive,
            negative,
            self._hashing.distance.rowwise,
            self._margin,
        )
        return loss
