from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer
from nlsh.learning.datasets import KNearestNeighborSiamese
from nlsh.learning.losses import contrastive_loss


class SiameseTrainer:

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

    def fit(self, K, batch_size=1024, learning_rate=3e-4, test_every_updates=300):
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

        dataset = KNearestNeighborSiamese(
            self._candidate_vectors_gpu,
            candidate_self_knn,
            k=100,
            positive_rate=self._positive_rate,
        )
        optimizer = torch.optim.Adam(
            self._hashing.parameters(),
            lr=learning_rate,
            amsgrad=True,
        )

        global_step = 0
        best_recall = 0.
        best_query_size = float("Inf")
        for _ in range(100):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1

                self._hashing.train_mode(True)
                optimizer.zero_grad()
                anchor = self._hashing.predict(sampled_batch[0])

                other = self._hashing.predict(sampled_batch[1])
                label = sampled_batch[2]
                loss = contrastive_loss(
                    anchor,
                    other,
                    label,
                    self._hashing.distance,
                    positive_margin=self._positive_margin,
                    negative_margin=self._negative_margin,
                )

                # TODO: potential uniform regularizers
                # loss += self._lambda1 * torch.mm(anchor, anchor.T).max(1)[0].mean()
                # loss -= self._lambda1 * torch.log(torch.cdist(anchor, anchor).topk(2, dim=1, largest=False)[0][:,1]).mean()

                self._logger.log("training/loss", loss.data.cpu(), global_step)
                loss.backward()
                optimizer.step()
                if global_step % test_every_updates == 0:
                    self._hashing.train_mode(False)
                    indexer = Indexer(
                        self._hashing,
                        self._candidate_vectors,
                        self._candidate_vectors_gpu,
                        self._data.distance,
                    )
                    n_indexes = len(indexer.index2row)
                    self._logger.log("test/n_indexes", n_indexes, global_step)
                    std_index_rows = np.std([len(idxs) for idxs in indexer.index2row.values()])
                    self._logger.log("test/std_index_rows", std_index_rows, global_step)

                    t1 = time()
                    recalls, n_candidates = indexer.query(self._validation_data, self._validation_data_gpu, k=K)
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
