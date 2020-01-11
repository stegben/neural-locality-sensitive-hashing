from time import time

import torch
import numpy as np

from .base import Trainer


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


class ProposedTrainer(Trainer):

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

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = KNearestNeighborAllOut(
            vectors,
            torch.tensor(self_knn, dtype=torch.long).cuda(),
            k=self._train_k,
        )
        return dataset

    def _get_loss(self, batch):
        batch_size = batch[0].shape[0]
        hashed_anchor = self._hashing.predict(batch[0])
        hashed_positives = self._hashing.predict(
            batch[1].view(-1, batch[0].shape[-1])
        ).view(batch_size, -1, hashed_anchor.shape[-1])
        n = self._candidate_vectors_gpu.shape[0]
        sampled_candidate_vectors = self._candidate_vectors_gpu[np.random.randint(0, n, (65536,)), :]
        hashed_candidates = self._hashing.predict(sampled_candidate_vectors)

        # knns should have smaller code distance
        positive_loss = self._hashing._distance_func.row_pairwise(
            hashed_anchor[:, None, :],
            hashed_positives,
        ).sum(dim=1).mean()

        query_index = self._hashing.hash(batch[0], n=1)
        query_index = [list(qi)[0] for qi in query_index]
        candidate_index = self._hashing.hash(sampled_candidate_vectors, n=1)
        candidate_index = [list(qi)[0] for qi in candidate_index]

        # query_mask = np.equal(*np.meshgrid(query_index, candidate_index, copy=False, indexing='ij'))
        # query_mask = torch.from_numpy(query_mask).float().cuda()
        # distances = self._hashing._distance_func.pairwise(
        #     hashed_anchor,
        #     hashed_candidates.detach(),
        # )
        # masked_distances = distances * query_mask
        # query_size_loss = torch.log(1e-10 + masked_distances).mean()
        # query_size_loss = masked_distances.sum(dim=1).mean()

        distances = torch.min(torch.abs(hashed_candidates - 0.5), axis=1)[0]
        distances = distances * torch.from_numpy(np.isin(candidate_index, query_index, invert=True)).cuda()
        query_size_loss = distances.sum()
        # sampled anchors should be as far from each other as possible
        loss = positive_loss + self._lambda1 * query_size_loss
        return loss
