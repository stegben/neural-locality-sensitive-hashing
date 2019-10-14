import os
from typing import List
import random
from functools import partial

from dotenv import load_dotenv
from tqdm import tqdm
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

from nlsh.networks import Encoder


load_dotenv()

DATA_PATH = os.environ.get("NLSH_GLOVE_25_PATH")
K = 10
HASH_SIZE = 12
WRITER = SummaryWriter(logdir=os.environ["NLSH_TENSORBOARD_LOG_DIR"])

class CandidateDataset(Dataset):

    def __init__(self, candidate_vectors, distance_func):
        self._candidate_vectors = candidate_vectors
        self._candidate_vectors_gpu = candidate_vectors.cuda()
        self._distance_func = distance_func
        self.n_candidates = self._candidate_vectors.shape[0]
        self.n_dims = self._candidate_vectors.shape[1]

    def __len__(self):
        return self.n_candidates

    def __getitem__(self, idx: int):
        v1_idx = random.randint(0, self.n_candidates-1)
        v2_idx = random.randint(0, self.n_candidates-1)
        anchor = self._candidate_vectors[idx, :]
        anchor_gpu = self._candidate_vectors_gpu[idx, :]
        v1 = self._candidate_vectors[v1_idx, :]
        v1_gpu = self._candidate_vectors_gpu[v1_idx, :]
        v2 = self._candidate_vectors[v2_idx, :]
        v2_gpu = self._candidate_vectors_gpu[v2_idx, :]
        d1 = 1 - self._distance_func(anchor, v1, dim=0)
        d2 = 1 - self._distance_func(anchor, v2, dim=0)
        if d1 > d2:
            return anchor_gpu, v2_gpu, v1_gpu
        return anchor_gpu, v1_gpu, v2_gpu



def calculate_recall(y_true, y_pred):
    # TODO: unittest
    n_true = len(y_true)
    true_positives = len(set(y_true) & set(y_pred))
    return true_positives / n_true


class NeuralLocalitySensitiveHashing:

    def __init__(self, encoder, distance_func):
        self._encoder = encoder
        self._distance_func = distance_func

    def fit(self, candidate_vectors, validation_data=None, ground_truth=None):
        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        validation_data = torch.from_numpy(validation_data).cuda()
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        dataset = CandidateDataset(self._candidate_vectors, self._distance_func)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(self._encoder.parameters())
        triplet_loss = nn.TripletMarginLoss(margin=0.0, p=2)

        global_step = 0
        for i_epoch in range(10):
            for i_batch, sampled_batch in enumerate(dataloader):
                global_step += 1
                optimizer.zero_grad()
                anchor = self._encoder(sampled_batch[0])
                positive = self._encoder(sampled_batch[1])
                negative = self._encoder(sampled_batch[2])
                loss = triplet_loss(anchor, positive, negative)
                WRITER.add_scalar("training loss", loss, global_step)
                loss.backward()
                optimizer.step()

                if i_batch % 10 == 0 and validation_data is not None:
                    self._build_index()
                    result = self.query(validation_data, k=100)
                    recall_scores = [calculate_recall(y_true, y_pred) for y_pred, y_true in zip(result, list(ground_truth))]
                    WRITER.add_scalar("test recall", np.mean(recall_scores), global_step)

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


def main():
    f_data = h5py.File(DATA_PATH)
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])

    encoder = Encoder(train_data.shape[1], HASH_SIZE).cuda()

    nlsh = NeuralLocalitySensitiveHashing(encoder, partial(F.cosine_similarity, dim=-1))

    nlsh.fit(train_data, test_data, ground_truth)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
