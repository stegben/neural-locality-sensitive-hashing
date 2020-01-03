import abc
from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer


class Trainer(abc.ABC):

    _data = None
    _hashing = None
    _logger = None
    _model_save_dir = "/tmp/"

    @abc.abstractmethod
    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        pass

    @abc.abstractmethod
    def _get_loss(self, batch):
        pass

    def _prepare_extra_models(self):
        pass

    def _get_extra_models_parameters(self):
        return []

    def fit(self, K, batch_size=1024, learning_rate=3e-4, test_every_updates=1000):
        if not self._data.prepared:
            self._data.load()
        candidate_vectors = self._data.training
        validation_data = self._data.testing
        candidate_self_knn = self._data.training_self_knn
        ground_truth = self._data.ground_truth[:, :K]

        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        self._validation_data = torch.from_numpy(validation_data)
        self._validation_data_gpu = self._validation_data.cuda()

        dataset = self._get_dataset(
            self._candidate_vectors_gpu,
            torch.from_numpy(candidate_self_knn).cuda(),
        )
        self._prepare_extra_models()

        optimizer = torch.optim.Adam(
            [p for p in self._hashing.parameters()] + self._get_extra_models_parameters(),
            lr=learning_rate,
            amsgrad=True,
        )

        global_step = 0
        best_recall = 0.
        best_query_size = float("Inf")

        for _ in range(300):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1

                self._hashing.train_mode(True)
                optimizer.zero_grad()

                loss = self._get_loss(sampled_batch)

                self._logger.log("training/loss", loss.data.cpu(), global_step)
                loss.backward()
                optimizer.step()
                if global_step % test_every_updates == 0:
                    self._hashing.train_mode(False)
                    indexer = Indexer(
                        self._hashing,
                        self._candidate_vectors_gpu,
                        self._data.distance,
                    )
                    n_indexes = len(indexer.index2row)
                    self._logger.log("test/n_indexes", n_indexes, global_step)
                    std_index_rows = np.std([len(idxs) for idxs in indexer.index2row.values()])
                    self._logger.log("test/std_index_rows", std_index_rows, global_step)

                    t1 = time()
                    recalls, n_candidates = indexer.query(self._validation_data_gpu, k=K)
                    t2 = time()
                    query_time = t2 - t1
                    current_recall = calculate_recall(list(ground_truth), recalls, np.mean)
                    current_query_size = np.mean(n_candidates)

                    if (current_recall > best_recall) and (current_query_size < best_query_size):
                        base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
                        self._hashing.save(base_name)
                        best_recall = current_recall

                    self._logger.log("test/recall", current_recall, global_step)
                    self._logger.log("test/query_size", current_query_size, global_step)
                    qps = self._validation_data.shape[0] / query_time
                    self._logger.log("test/qps", qps, global_step)
