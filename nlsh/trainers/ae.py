from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import Trainer


class OnePass:

    def __init__(self, candidate_vectors):
        self._candidate_vectors = candidate_vectors
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

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


class Decoder(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, self.output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class AE(Trainer):

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = OnePass(vectors)
        return dataset

    def _get_loss(self, batch):
        probs = self._hashing.predict(batch)
        reconstruct = self.decoder(probs)
        dist = self._data.distance(reconstruct, batch)
        loss = (dist**2).mean()
        return loss

    def _prepare_extra_models(self):
        self.decoder = Decoder(
            input_dim=self._hashing.output_dim,
            output_dim=self._data.dim,
        ).cuda()

    def _get_extra_models_parameters(self):
        return [p for p in self.decoder.parameters()]
