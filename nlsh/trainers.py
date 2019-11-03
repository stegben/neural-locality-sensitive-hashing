import abc
from time import time

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from nlsh.loggers import NullLogger
from nlsh.learning.datasets import KNearestNeighborTriplet
from nlsh.learning.losses import triplet_loss


def calculate_recall(y_true, y_pred):
    # TODO: unittest
    n_true = len(y_true)
    true_positives = len(set(y_true) & set(y_pred))
    return true_positives / n_true


class _EvalDataset(Dataset):

    def __init__(self, vector):
        self._vector = vector

    def __len__(self):
        return self._vector.shape[0]

    def __getitem__(self, idx):
        return self._vector[idx, :]


class TripletTrainer:

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger=None,
            lambda1=0.001,
            triplet_margin=0.1,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger or NullLogger()
        self._lambda1 = lambda1
        self._triplet_margin = triplet_margin

    def fit(self, K, batch_size=1024, learning_rate=3e-4, test_every_updates=1000):
        if not self._data.prepared:
            self._data.load()
        candidate_vectors = self._data.training
        validation_data = self._data.testing
        candidate_self_knn = self._data.training_self_knn
        ground_truth = self._data.ground_truth[:, :K]

        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        self._validation_data = torch.from_numpy(validation_data)
        self._validation_data_gpu = self._validation_data.cuda()

        dataset = KNearestNeighborTriplet(
            self._candidate_vectors_gpu,
            candidate_self_knn,
            k=100,
        )
        optimizer = torch.optim.Adam(
            self._hashing.parameters(),
            lr=learning_rate,
            amsgrad=True,
        )

        global_step = 0
        best_recall = 0.
        for _ in range(10000):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1
                optimizer.zero_grad()
                anchor = self._hashing.predict(sampled_batch[0])
                positive = self._hashing.predict(sampled_batch[1])
                negative = self._hashing.predict(sampled_batch[2])
                loss = triplet_loss(
                    anchor,
                    positive,
                    negative,
                    self._hashing.distance,
                    self._triplet_margin,
                )
                loss += self._lambda1  * ((0.5 - anchor.mean(0))**2).mean()
                self._logger.log("training/loss", loss, global_step)
                loss.backward()
                optimizer.step()
                if global_step % test_every_updates == 0:

                    self._build_index()

                    t1 = time()
                    result = self.query(self._validation_data_gpu, k=K)
                    t2 = time()
                    query_time = t2 - t1
                    current_recall = np.mean([
                        calculate_recall(y_true, y_pred)
                        for y_pred, y_true in zip(result, list(ground_truth))
                    ])

                    if current_recall > best_recall:
                        base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
                        self._hashing.save(base_name)
                        best_recall = current_recall

                    self._logger.log("test/recall", current_recall, global_step)

                    n_indexes = len(self.index2row)
                    self._logger.log("test/n_indexes", n_indexes, global_step)
                    std_index_rows = np.std([len(idxs) for idxs in self.index2row.values()])
                    self._logger.log("test/std_index_rows", std_index_rows, global_step)
                    qps = self._validation_data.shape[0] / query_time
                    self._logger.log("test/qps", qps, global_step)

            # NOTE: possible regularize method: anchor.pow(2) - 1 (more confident)
        self._build_index()

    def _build_index(self):
        indexes = self.hash(self._candidate_vectors_gpu)
        self.index2row = {}
        for idx, index in enumerate(indexes):
            if index not in self.index2row:
                self.index2row[index] = [idx]
            else:
                self.index2row[index].append(idx)

        # NOTE: this is a import speed optimization
        # allocating a new LongTensor is non trivial and will dominate
        # the evaluation process time
        for index, rows in self.index2row.items():
            self.index2row[index] = torch.LongTensor(rows)

    def hash(self, query_vectors, batch_size=1024):
        hash_keys = []

        n = query_vectors.shape[0]
        n_batches = n // batch_size
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = query_vectors[start:end, :]
            hash_key = self._hashing.hash(batch)
            hash_keys += hash_key
        last_batch = query_vectors[n_batches*batch_size:, :]
        hash_key = self._hashing.hash(last_batch)
        hash_keys += hash_key
        return hash_keys

    def query(self, query_vectors, k=10):
        query_indexes = self.hash(query_vectors)
        result = []
        vector_buffer = torch.rand(self._candidate_vectors.shape)
        for idx, qi in enumerate(query_indexes):
            candidate_rows = self.index2row.get(qi, torch.LongTensor([]))

            n_candidates = len(candidate_rows)
            target_vector = self._validation_data[idx, :]

            # NOTE: indexing with tensor will create a copy
            # use index_select will directly move data from one to
            # another. This highly reduce the memory allocation overhead
            torch.index_select(
                self._candidate_vectors,
                0,
                candidate_rows,
                out=vector_buffer[:n_candidates, :],
            )
            distance = self._data.distance(
                target_vector,
                vector_buffer[:n_candidates, :],
            )
            try:
                topk_idxs = distance.topk(k, largest=False)[1].tolist()
                topk_idxs = [int(candidate_rows[i]) for i in topk_idxs]
            except RuntimeError:
                topk_idxs = candidate_rows

            result.append(topk_idxs)
        return result


