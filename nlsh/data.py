from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import numpy as np


def norm_to_unit_sphere(arr):
    return arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]


# TODO: unify member type and pairwise_distance library
class Glove:

    def __init__(self, path, normalized=False):
        self.f = h5py.File(path, "r")
        self._normalized = normalized
        # TODO: retry 3 times, wait for 5 sec each time
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

        if self._normalized:
            self._training = norm_to_unit_sphere(self._training)
            self._testing = norm_to_unit_sphere(self._testing)

        self.f.close()

        self._dim = self._training.shape[1]
        self._prepared = True

    def _check_prepared(self):
        if not self._prepared:
            raise ValueError(f"{self.__class__.__name__} is not prepared. call `load` beforehand.")

    @property
    def prepared(self):
        return self._prepared

    @property
    def dim(self):
        self._check_prepared()
        return self._dim

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

    @staticmethod
    def distance(v1, v2):
        """Cosine distance betwenn 2 matrix

        v1: (d)
        v2: (n, d)

        Returns
        D: (n) where D[i] is the distance between v1 and v2[i, :]
        """
        return 1 - F.cosine_similarity(v1, v2, dim=-1)

# TODO:
class SIFT:
    pass


class BigANN1B:
    pass


class Deep1B:
    pass
