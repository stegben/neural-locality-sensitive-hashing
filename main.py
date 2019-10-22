import os
from typing import List
from datetime import datetime
import random

from dotenv import load_dotenv
from tqdm import tqdm
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter

from nlsh.networks import Encoder


load_dotenv()

DATA_PATH = os.environ.get("NLSH_PROCESSED_GLOVE_25_PATH")
K = 10
HASH_SIZE = 12
LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
RUN_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITER = SummaryWriter(logdir=f"{LOG_BASE_DIR}/{RUN_NAME}")
LAMBDA1 = 1e-2


def calculate_recall(y_true, y_pred):
    # TODO: unittest
    n_true = len(y_true)
    true_positives = len(set(y_true) & set(y_pred))
    return true_positives / n_true

class KNearestNeighborPositive(Dataset):

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

    def __getitem__(self, idx: int):
        anchor = self._candidate_vectors[idx, :]

        # positive sample from _candidate_self_knn
        pre_v1_idx = random.randint(0, self.k - 1)
        v1_idx = self._candidate_self_knn[idx, pre_v1_idx]

        # negative sample randomly select from the dataset
        v2_idx = random.randint(0, self.n_candidates - 1)

        v1 = self._candidate_vectors[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        return anchor, v1, v2

class NeuralLocalitySensitiveHashing:

    def __init__(self, encoder, distance_func):
        self._encoder = encoder
        self._distance_func = distance_func

    def fit(
            self,
            candidate_vectors,
            candidate_self_knn,
            validation_data=None,
            ground_truth=None,
        ):
        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        validation_data = torch.from_numpy(validation_data)
        validation_data_gpu = validation_data.cuda()
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        print("Pre-calculate train/test distance")
        self._precalculate_distances = self._distance_func(
            validation_data,
            self._candidate_vectors,
        )
        dataset = KNearestNeighborPositive(
            self._candidate_vectors,
            candidate_self_knn,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(
            self._encoder.parameters(),
            lr=1e-4,
            amsgrad=True,
        )
        triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)

        global_step = 0
        for _ in range(100):
            for i_batch, sampled_batch in enumerate(dataloader):
                global_step += 1
                optimizer.zero_grad()
                anchor = self._encoder(sampled_batch[0].cuda())
                positive = self._encoder(sampled_batch[1].cuda())
                negative = self._encoder(sampled_batch[2].cuda())
                loss = triplet_loss(anchor, positive, negative) \
                       + LAMBDA1 * ((0.5 - anchor.mean(0))**2).mean()
                WRITER.add_scalar("training loss", loss, global_step)
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
                    recall_scores = np.mean([
                        calculate_recall(y_true, y_pred)
                        for y_pred, y_true in zip(result, list(ground_truth))
                    ])
                    # pr.disable()
                    # pstats.Stats(pr).sort_stats('tottime').print_stats(5)
                    # import ipdb; ipdb.set_trace()
                    WRITER.add_scalar("test recall", recall_scores, global_step)

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
        prob = self._encoder(query_vectors)
        codes = (prob > 0.5).int().tolist()
        return [''.join(str(c) for c in binarr) for binarr in codes]
        # TODO this hash function time dominate at later epochs

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


def pairwise_cosine_distance(v1, v2, dim=-1):
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


def main():
    f_data = h5py.File(DATA_PATH)
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])[:, :K]
    train_knn = np.array(f_data['train_knn'])
    f_data.close()
    encoder = Encoder(train_data.shape[1], HASH_SIZE).cuda()

    nlsh = NeuralLocalitySensitiveHashing(encoder, pairwise_cosine_distance)

    nlsh.fit(train_data, train_knn, test_data, ground_truth)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
