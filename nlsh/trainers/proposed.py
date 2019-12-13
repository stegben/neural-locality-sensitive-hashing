from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer


class KNearestNeighborAllOut:

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

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        positives = torch.zeros(
            (batch_size, self.k, self._candidate_vectors.shape[1]),
            dtype=self._candidate_vectors.dtype,
            device=self._candidate_vectors.device,
        )
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            selected_idxs = anchor_idxs[start:end]
            anchor = self._candidate_vectors[selected_idxs, :]

            for batch_inner_idx in range(batch_size):
                torch.index_select(
                    self._candidate_vectors,
                    0,
                    self._candidate_self_knn[selected_idxs[batch_inner_idx], :self.k],
                    out=positives[batch_inner_idx, :, :],
                )
            yield anchor, positives


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
        # qs_mask = (1. - torch.eye(batch_size, dtype=torch.float32)
        n = self._candidate_vectors.shape[0]
        for _ in range(100):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1

                self._hashing.train_mode(True)
                optimizer.zero_grad()

                hashed_anchor = self._hashing.predict(sampled_batch[0])
                hashed_positives = self._hashing.predict(
                    sampled_batch[1].view(-1, sampled_batch[0].shape[-1])
                ).view(batch_size, -1, hashed_anchor.shape[-1])

                sampled_candidate_vectors = self._candidate_vectors_gpu[np.random.randint(0, n, (4096,)), :]
                hashed_candidates = self._hashing.predict(sampled_candidate_vectors)

                positive_loss = self._hashing._distance_func.row_pairwise(
                    hashed_anchor[:, None, :],
                    hashed_positives,
                ).sum(dim=1).mean()
                # positive_loss = 0
                # knns should have smaller code distance

                query_index = self._hashing.hash(sampled_batch[0])
                candidate_index = self._hashing.hash(sampled_candidate_vectors)
                query_mask = np.equal(*np.meshgrid(query_index, candidate_index, copy=False, indexing='ij'))
                query_mask = torch.from_numpy(query_mask).float().cuda()
                distances = self._hashing._distance_func.pairwise(
                    hashed_anchor.detach(),
                    hashed_candidates,
                )
                masked_distances = distances * query_mask
                # query_size_loss = torch.log(1e-10 + masked_distances).mean()
                query_size_loss = masked_distances.sum(dim=1).mean()
                # sampled anchors should be as far from each other as possible
                loss = positive_loss - self._lambda1 * query_size_loss

                self._logger.log("training/loss", loss.data.cpu(), global_step)
                self._logger.log("training/positive_loss", positive_loss.data.cpu(), global_step)
                self._logger.log("training/query_size_loss", query_size_loss.data.cpu(), global_step)
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
