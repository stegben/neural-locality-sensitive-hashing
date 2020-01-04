from time import time
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

from .base import Trainer

NSM_RANDOM = "random"
NSM_NEAREST = "nearest"  # get the closest negative sample
NSM_HARD = "hard"  # sample from all Dn < max(Dp)
NSM_SEMI_HARD = "semi-hard"  # sample from Dn < (max(Dp) + margin) & Dn > max(Dp)


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


def batch_execute(batchable, execution, batch_size, concatenator):
    batch_results = []
    n = batchable.shape[0]
    n_batches = n // batch_size
    with torch.no_grad():
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = batchable[start:end, :]
            batch_results.append(execution(batch))
        last_batch = batchable[n_batches*batch_size:, :]
        batch_results.append(execution(last_batch))
    return concatenator(batch_results)


def nearest_exclude_positive(vectors, distance_function, positive_indexes):
    batch_results = []

    batch_size = 8
    n = vectors.shape[0]
    n_batches = n // batch_size

    with torch.no_grad():
        diagnal_idx_for_scatter = torch.arange(batch_size).reshape(batch_size, 1).cuda()
        for idx in tqdm(range(n_batches)):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = vectors[start:end, :]
            distances = distance_function(batch, vectors)
            max_value = distances.max()
            distances.scatter_(1, positive_indexes[start:end, :], max_value)  # give positive indexes max value so that we don't choose them
            distances.scatter_(1, diagnal_idx_for_scatter, max_value)  # don't select self
            nearest_negative_sample_idx = distances.argmin(dim=1)
            batch_results.append(nearest_negative_sample_idx)

        start = n_batches * batch_size
        diagnal_idx_for_scatter = torch.arange(n - start + 1).reshape(batch_size, 1).cuda()
        last_batch = vectors[start:, :]
        distances = distance_function(last_batch, vectors)
        distances.scatter_(1, positive_indexes[start:, :], distances.max())
        distances.scatter_(1, diagnal_idx_for_scatter, distances.max())
        nearest_negative_sample_idx = distances.argmin(dim=1)
        batch_results.append(nearest_negative_sample_idx)

    return torch.cat(batch_results)


class KNearestNeighborTriplet:

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            k=None,
            encoder=None,
            distance_func=None,
            negative_sampling_method="random",
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn

        self.k = k or candidate_self_knn.shape[1]
        self.n_candidates = self._candidate_vectors.shape[0]

        self.encoder = encoder
        self.distance_func = distance_func
        self.negative_sampling_method = negative_sampling_method

    def __len__(self):
        return self.n_candidates

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        if shuffle:
            anchor_idxs = torch.randperm(len(self)).cuda()
        else:
            anchor_idxs = torch.arange(len(self)).cuda()
        knn_col_idxs = torch.randint(0, self.k, (len(self),))

        if self.negative_sampling_method == NSM_RANDOM:
            negative_idxs = torch.randint(0, len(self), (len(self),))
        elif self.negative_sampling_method == NSM_NEAREST:
            print("prepare new negative_idxs")
            candidate_encodings = batch_execute(self._candidate_vectors, self.encoder, 128, partial(torch.cat, dim=0))
            negative_idxs = nearest_exclude_positive(
                candidate_encodings,
                self.distance_func,
                self._candidate_self_knn[:, :self.k],
            )

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            knn_idxs = self._candidate_self_knn[anchor_idxs[start:end], knn_col_idxs[start:end]]
            positive = self._candidate_vectors[knn_idxs, :]

            negative = self._candidate_vectors[negative_idxs[start:end], :]
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
            positive_k=None,
            negative_sampling_method='random',
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger
        self._lambda1 = lambda1
        self._margin = margin
        self._positive_k = positive_k
        self._negative_sampling_method = negative_sampling_method

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = KNearestNeighborTriplet(
            vectors,
            self_knn,
            k=self._positive_k,
            negative_sampling_method=self._negative_sampling_method,
            distance_func=self._hashing._distance_func.pairwise,
            encoder=self._hashing._hasher,
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
