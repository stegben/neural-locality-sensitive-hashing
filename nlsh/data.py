from pathlib import Path

import h5py
import torch
import numpy as np


class Glove:

    def __init__(self, path):
        self.f = h5py.File(path, "r")
        self._prepared = False

    def load(self):
        self._training = np.array(self.f['train'])
        self._testing = np.array(self.f['test'])
        self._ground_truth = np.array(self.f['neighbors'])
        try:
            self._training_self_knn = np.array(self.f['train_knn'])
        except KeyError:
            # TODO: precompute knn here. For now, call `python precompute.py {data_name}`
            pass
        self.f.close()
        self._prepared = True

    def _check_prepared(self):
        if not self._prepared:
            raise ValueError(f"{self.__class__.__name__} is not prepared. call `load` beforehand.")

    @property
    def training(self):
        self._check_prepared()
        return self._training

    @property
    def testing(self):
        self._check_prepared()
        return self._testing

    @property
    def ground_truth(self):
        self._check_prepared()
        return self._ground_truth

    @property
    def training_self_knn(self):
        self._check_prepared()
        return self._training_self_knn

    @staticmethod
    def pairwise_distance(v1, v2):
        """Cosine distance betwenn 2 matrix

        v1: (n, d)
        v2: (m, d)

        Returns
        D: (n, m) where D[i, j] is the distance between v1[i, :] and v2[j, :]
        """
        v1_normalized = v1 / v1.norm(dim=1)[:, None]
        v2_normalized = v2 / v2.norm(dim=1)[:, None]
        cosine_similarity = torch.mm(v1_normalized, v2_normalized.T)
        return 1 - cosine_similarity


class SIFT:
    pass
