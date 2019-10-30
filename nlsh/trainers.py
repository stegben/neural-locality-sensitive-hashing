import abc

import torch
from torch.utils.data import DataLoader
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
        ground_truth = self._data.ground_truth

        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        validation_data = torch.from_numpy(validation_data)
        validation_data_gpu = validation_data.cuda()
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()

        # TODO: test if we need to precompute this. It takes 40 GB memory
        print("Pre-calculate train/test distance")
        self._precalculate_distances = self._data.pairwise_distance(
            validation_data,
            self._candidate_vectors,
        )
        dataset = KNearestNeighborTriplet(
            self._candidate_vectors,
            candidate_self_knn,
            k=100,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
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
                anchor = self._hashing.predict(sampled_batch[0].cuda())
                positive = self._hashing.predict(sampled_batch[1].cuda())
                negative = self._hashing.predict(sampled_batch[2].cuda())
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
                # 715 MB
                if i_batch % 100 == 0:
                    # import cProfile, pstats, io
                    # pr = cProfile.Profile()
                    # pr.enable()

                    self._build_index()
                    # 5500Mb
                    # TODO: fix the memory usage

                    result = self.query(validation_data_gpu, k=K)
                    current_recall = np.mean([
                        calculate_recall(y_true, y_pred)
                        for y_pred, y_true in zip(result, list(ground_truth))
                    ])

                    if current_recall > best_recall:
                        base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
                        self._hashing.save(base_name)
                        best_recall = current_recall
                    # pr.disable()
                    # pstats.Stats(pr).sort_stats('tottime').print_stats(5)
                    # import ipdb; ipdb.set_trace()
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
        return self._hashing.hash(query_vectors)

    def query(self, query_vectors, k=10):
        query_indexes = self.hash(query_vectors)
        result = []
        for idx, qi in enumerate(tqdm(query_indexes)):
            candidate_rows = self.index2row.get(qi, [])
            # TODO: too slow, possibly because create different size vector every loop
            topk_idxs = self._precalculate_distances[idx, candidate_rows].argsort()[:k]
            topk_idxs = [candidate_rows[i] for i in topk_idxs]
            result.append(topk_idxs)
        return result
