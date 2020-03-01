from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import numpy as np


def norm_to_unit_sphere(arr):
    return arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]


# TODO: unify member type and pairwise_distance library
class Glove:

    def __init__(self, path, unit_norm=False, unit_ball=False):
        self.f = h5py.File(path, "r")
        self._unit_norm = unit_norm
        self._unit_ball = unit_ball
        # TODO: retry 3 times, wait for 5 sec each time
        self._prepared = False

    def load(self):
        print("reading glove training data...")
        self._training = np.array(self.f['train'])
        print("reading glove testing data...")
        self._testing = np.array(self.f['test'])

        if self._unit_norm:
            mean = self._training.mean(0)
            std = self._training.std(0)
            self._training = (self._training - mean) / std
            self._testing = (self._testing - mean) / std

        if self._unit_ball:
            self._training = norm_to_unit_sphere(self._training)
            self._testing = norm_to_unit_sphere(self._testing)

        self._ground_truth = np.array(self.f['neighbors'])
        try:
            self._training_self_knn = np.array(self.f['train_knn'])
        except KeyError:
            # TODO: precompute knn here. For now, call `python precompute.py {data_name}`
            pass

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
    def __init__(self, path, unit_norm=False):
        self.f = h5py.File(path, "r")
        self._unit_norm = unit_norm
        # TODO: retry 3 times, wait for 5 sec each time
        self._prepared = False

    def load(self):
        print("reading glove training data...")
        self._training = np.array(self.f['train'])
        print("reading glove testing data...")
        self._testing = np.array(self.f['test'])

        if self._unit_norm:
            mean = self._training.mean(0)
            std = self._training.std(0)
            self._training = (self._training - mean) / std
            self._testing = (self._testing - mean) / std

        self._ground_truth = np.array(self.f['neighbors'])
        try:
            self._training_self_knn = np.array(self.f['train_knn'])
        except KeyError:
            # TODO: precompute knn here. For now, call `python precompute.py {data_name}`
            pass

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
        """Euclidean distance betwenn 2 matrix

        v1: (n, d)
        v2: (m, d)

        Returns
        D: (n, m) where D[i, j] is the distance between v1[i, :] and v2[j, :]
        """
        p_norm = p.pow(2).sum(dim=-1, keepdim=True)
        q_norm = q.pow(2).sum(dim=-1, keepdim=True)
        result = torch.addmm(q_norm.transpose(-2, -1), p, q.transpose(-2, -1), alpha=-2).add_(p_norm)
        return torch.sqrt(result)

    @staticmethod
    def distance(v1, v2):
        """Euclidean distance betwenn 2 matrix

        v1: (d)
        v2: (n, d)

        Returns
        D: (n) where D[i] is the distance between v1 and v2[i, :]
        """
        return F.pairwise_distance(v1, v2)


class BigANN1B:
    pass


class Deep1B:
    pass
