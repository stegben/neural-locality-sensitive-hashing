from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer


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


class KNearestNeighborTriplet:

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

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            knn_idxs = self._candidate_self_knn[anchor_idxs[start:end], np.random.randint(0, self.k, (batch_size,))]
            positive = self._candidate_vectors[knn_idxs, :]

            random_idxs = np.random.randint(0, len(self), (batch_size,))
            negative = self._candidate_vectors[random_idxs, :]
            yield anchor, positive, negative

class TripletTrainer:

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
            lambda1=0.001,
            margin=0.1,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger
        self._lambda1 = lambda1
        self._margin = margin

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
        best_query_size = float("Inf")
        for _ in range(300):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1

                self._hashing.train_mode(True)
                optimizer.zero_grad()
                anchor = self._hashing.predict(sampled_batch[0])
                positive = self._hashing.predict(sampled_batch[1])
                negative = self._hashing.predict(sampled_batch[2])
                loss = triplet_loss(
                    anchor,
                    positive,
                    negative,
                    self._hashing.distance.rowwise,
                    self._margin,
                )

                # TODO: DI uniform regularizers
                # loss += self._lambda1 * torch.mm(anchor, anchor.T).max(1)[0].mean()
                # loss -= self._lambda1 * torch.log(torch.cdist(anchor, anchor).topk(2, dim=1, largest=False)[0][:,1]).mean()

                self._logger.log("training/loss", loss.data.cpu(), global_step)
                loss.backward()
                optimizer.step()
                if global_step % test_every_updates == 0:
                    self._hashing.train_mode(False)
                    # import ipdb; ipdb.set_trace()
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
