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
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from nlsh.networks import Encoder
from nlsh.datasets import RandomPositive


load_dotenv()

DATA_PATH = os.environ.get("NLSH_GLOVE_25_PATH")
K = 10
HASH_SIZE = 12
LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
RUN_NAME = datetime.now().strftime("%Y%m%d-%H%M%S")
WRITER = SummaryWriter(logdir=f"{LOG_BASE_DIR}/{RUN_NAME}")
LAMBDA1 = 1e-1


def calculate_recall(y_true, y_pred):
    # TODO: unittest
    n_true = len(y_true)
    true_positives = len(set(y_true) & set(y_pred))
    return true_positives / n_true


class NeuralLocalitySensitiveHashing:

    def __init__(self, encoder, dataset ,distance_func):
        self._encoder = encoder
        self._distance_func = distance_func

    def fit(self, candidate_vectors, validation_data=None, ground_truth=None):
        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        validation_data = torch.from_numpy(validation_data).cuda()
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        dataset = RandomPositive(self._candidate_vectors, self._distance_func)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(
            self._encoder.parameters(),
            lr=1e-5,
            amsgrad=True,
        )
        triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)

        global_step = 0
        for i_epoch in range(10):
            for i_batch, sampled_batch in enumerate(dataloader):
                global_step += 1
                optimizer.zero_grad()
                anchor = self._encoder(sampled_batch[0])
                positive = self._encoder(sampled_batch[1])
                negative = self._encoder(sampled_batch[2])
                loss = triplet_loss(anchor, positive, negative) \
                       + LAMBDA1 * ((0.5 - anchor.mean(0))**2).mean()
                WRITER.add_scalar("training loss", loss, global_step)
                loss.backward()
                optimizer.step()

                if i_batch % 10 == 0 and validation_data is not None:
                    self._build_index()
                    result = self.query(validation_data, k=K)
                    recall_scores = np.mean([
                        calculate_recall(y_true, y_pred)
                        for y_pred, y_true in zip(result, list(ground_truth))
                    ])
                    WRITER.add_scalar("test recall", recall_scores, global_step)

                # regularize 1: anchor.sum(axis=1) - 0.5 (more uniform)
                # regularize 2: anchor.pow(2) - 1 (more confident)
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

    def query(self, query_vectors, k=10):
        query_indexes = self.hash(query_vectors)
        result = []
        for idx, qi in enumerate(tqdm(query_indexes)):
            query_vector = query_vectors[idx, :]
            candidate_rows = self.index2row.get(qi, [])
            topk_idxs = self._brutal_distance_sort(candidate_rows, query_vector)[:k]
            result.append(topk_idxs)
        return result

    def _brutal_distance_sort(self, candidate_rows, query_vector) -> List[int]:
        # TODO: unittest
        candidate_vectors = self._candidate_vectors_gpu[candidate_rows, :]
        distance = self._distance_func(candidate_vectors, query_vector, dim=-1)
        return [candidate_rows[idx] for idx in distance.argsort().tolist()]


def cosine_distance(v1, v2, dim=-1):
    return 1 - F.cosine_similarity(v1, v2, dim=dim)


def main():
    f_data = h5py.File(DATA_PATH)
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])[:, :K]
    import ipdb; ipdb.set_trace()
    encoder = Encoder(train_data.shape[1], HASH_SIZE).cuda()

    nlsh = NeuralLocalitySensitiveHashing(encoder, cosine_distance)

    nlsh.fit(train_data, test_data, ground_truth)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
