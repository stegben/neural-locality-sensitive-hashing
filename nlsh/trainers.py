import abc

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

    def fit(self, K):
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

        # # TODO: test if we need to precompute this. It takes 40 GB memory
        # print("Pre-calculate train/test distance")
        # self._precalculate_distances = self._data.pairwise_distance(
        #     validation_data,
        #     self._candidate_vectors,
        # )
        dataset = KNearestNeighborTriplet(
            self._candidate_vectors_gpu,
            candidate_self_knn,
            k=100,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=0,
            # pin_memory=True,
        )
        optimizer = torch.optim.Adam(
            self._hashing.parameters(),
            lr=3e-4,
            amsgrad=True,
        )

        global_step = 0
        best_recall = 0.
        for _ in range(10000):
            for i_batch, sampled_batch in enumerate(dataloader):
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
                if i_batch % 100 == 0:
                    import cProfile, pstats, io
                    pr = cProfile.Profile()
                    pr.enable()

                    self._build_index()

                    result = self.query(self._validation_data_gpu, k=K)
                    current_recall = np.mean([
                        calculate_recall(y_true, y_pred)
                        for y_pred, y_true in zip(result, list(ground_truth))
                    ])

                    if current_recall > best_recall:
                        base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
                        self._hashing.save(base_name)
                        best_recall = current_recall
                    pr.disable()
                    pstats.Stats(pr).sort_stats('tottime').print_stats(5)
                    import ipdb; ipdb.set_trace()
                    self._logger.log("test/recall", current_recall, global_step)

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
        distribution = [len(idxs) for idxs in self.index2row.values()]
        print(f"Create {len(self.index2row)} indexes, with std {np.std(distribution)}")

    def hash(self, query_vectors):
        # eval_dataset = _EvalDataset(query_vectors)
        # dl = DataLoader(
        #     eval_dataset,
        #     batch_size=1024,
        #     shuffle=False,
        #     num_workers=0,
        # )
        hash_keys = []

        n = query_vectors.shape[0]
        batch_size = 1024
        n_batches = n // batch_size
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = query_vectors[start:end, :]
        # for batch in dl:
            hash_key = self._hashing.hash(batch)
            hash_keys += hash_key
        last_batch = query_vectors[n_batches*batch_size:, :]
        hash_key = self._hashing.hash(last_batch)
        hash_keys += hash_key
        return hash_keys

    def query(self, query_vectors, k=10):
        query_indexes = self.hash(query_vectors)
        result = []
        distance_buffer = torch.rand(self._candidate_vectors.shape[0])
        for idx, qi in enumerate(tqdm(query_indexes)):
            candidate_rows = self.index2row.get(qi, [])
            # TODO: too slow, possibly because create different size vector every loop
            topk_idxs = self.get_top_k_index(idx, candidate_rows, k, distance_buffer)
            # topk_idxs = [candidate_rows[i] for i in topk_idxs]
            result.append(topk_idxs)
        return result

    def get_top_k_index(self, idx, candidate_rows, k, distance_buffer):
        # try 1
        # return self._precalculate_distances[idx, candidate_rows].argsort()[:k]

        # try 2
        # try:
        #     topk_idxs = self._precalculate_distances[idx, candidate_rows].topk(k, largest=False)[1].tolist()
        #     return [candidate_rows[i] for i in topk_idxs]
        # except RuntimeError:
        #     return candidate_rows

        # try 3
        try:
            n_candidates = len(candidate_rows)
            target_vector = self._validation_data[[idx], :]
            # candidate_vectors = self._candidate_vectors.select(0, candidate_rows)
            # distance = self._data.pairwise_distance(target_vector, candidate_vectors)[:0]
            import torch.nn.functional as F
            distance_buffer[:n_candidates] = F.cosine_similarity(
                target_vector,
                self._candidate_vectors[:n_candidates, :],
                dim=-1,
            )
            distance_buffer[:n_candidates].mul_(-1).add_(1)
            topk_idxs = distance_buffer[:n_candidates].topk(k, largest=False)[1].tolist()
            return [candidate_rows[i] for i in topk_idxs]
        except RuntimeError:
            return candidate_rows
