from time import time

import torch
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer


def contrastive_loss(
        anchor,
        other,
        label,
        distance_func,
        negative_margin=0.1,
        positive_margin=0.0,
    ):
    d = distance_func(anchor, other)
    positive_loss = label * torch.clamp(d - positive_margin, min=0)**2
    negative_loss = (1 - label) * torch.clamp(d - negative_margin, max=0)**2
    return torch.mean(positive_loss + negative_loss) / 2


class KNearestNeighborSiamese:

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            k=None,
            positive_rate=0.1,
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn
        self._positive_rate = positive_rate

        self.k = k or candidate_self_knn.shape[1]
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        label_all_np = np.random.random((len(self),)) < self._positive_rate
        label_all = torch.from_numpy(label_all_np).long().cuda()

        negative_idx_all = np.random.randint(0, len(self), (len(self),))
        # negative_idx_all = torch.from_numpy(negative_idx_all).long().cuda()

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            ramdomly_selected_positive_idx = np.random.randint(0, self.k, (batch_size,))
            positive_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_positive_idx]
            negative_idxs = negative_idx_all[start:end]
            label = label_all_np[start:end]
            label_output = label_all[start:end]
            other_idx = positive_idxs * label + negative_idxs * (1 - label)
            other = self._candidate_vectors[other_idx, :]
            yield anchor, other, label_output


class KNearestNeighborLocallySiamese:

    def __init__(
            self,
            candidate_vectors,
            candidate_self_knn,
            inner_k=None,
            outer_k=None,
            positive_rate=0.1,
        ):
        self._candidate_vectors = candidate_vectors
        self._candidate_self_knn = candidate_self_knn
        self._positive_rate = positive_rate

        self._inner_k = inner_k or candidate_self_knn.shape[1] // 2
        self._outer_k = outer_k or candidate_self_knn.shape[1]
        if self._outer_k <= self._inner_k:
            raise ValueError(f"Outer K (got {self._outer_k}) should be larger than inner K (got {self._inner_k}).")
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def batch_generator(self, batch_size, shuffle=False):
        n_batches = len(self) // batch_size

        label_all_np = np.random.random((len(self),)) < self._positive_rate
        label_all = torch.from_numpy(label_all_np).long().cuda()

        anchor_idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(anchor_idxs)

        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size

            anchor = self._candidate_vectors[anchor_idxs[start:end], :]

            ramdomly_selected_positive_idx = np.random.randint(0, self._inner_k, (batch_size,))
            ramdomly_selected_negative_idx = np.random.randint(self._inner_k, self._outer_k, (batch_size,))
            positive_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_positive_idx]
            negative_idxs = self._candidate_self_knn[anchor_idxs[start:end], ramdomly_selected_negative_idx]
            label = label_all_np[start:end]
            label_output = label_all[start:end]
            other_idx = positive_idxs * label + negative_idxs * (1 - label)
            other = self._candidate_vectors[other_idx, :]
            yield anchor, other, label_output


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
                    self._hashing.distance.rowwise,
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
