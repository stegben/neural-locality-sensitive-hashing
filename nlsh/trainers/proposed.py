from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer
from nlsh.learning.datasets import KNearestNeighborAllOut
from nlsh.learning.losses import contrastive_loss


class ProposedTrainer:

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
            train_k=10,
            lambda1=0.001,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger
        self._train_k = train_k
        self._lambda1 = lambda1

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

        dataset = KNearestNeighborAllOut(
            self._candidate_vectors_gpu,
            torch.tensor(candidate_self_knn, dtype=torch.long).cuda(),
            k=self._train_k,
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

                hashed_anchor = self._hashing.predict(sampled_batch[0])
                hashed_positives = self._hashing.predict(
                    sampled_batch[1].view(-1, sampled_batch[0].shape[-1])
                ).view(batch_size, -1, hashed_anchor.shape[-1])

                positive_loss = self._hashing._distance_func.row_pairwise(
                    hashed_anchor[:, None, :],
                    hashed_positives,
                ).mean()
                # knns should have smaller code distance

                query_size_loss = torch.log(1 + self._hashing._distance_func.pairwise(
                    hashed_anchor,
                    hashed_anchor,
                )).mean()
                # sampled anchors should be as far from each other as possible
                loss = positive_loss - self._lambda1 * query_size_loss

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
