from typing import List

import torch


def build_index(indexes):
    index2row = {}
    for idx, index in enumerate(indexes):
        if index not in index2row:
            index2row[index] = [idx]
        else:
            index2row[index].append(idx)

    # NOTE: this is a import speed optimization
    # allocating a new LongTensor is non trivial and will dominate
    # the evaluation process time
    for index, rows in index2row.items():
        index2row[index] = torch.LongTensor(rows).cuda()

    return index2row


class Indexer:

    def __init__(self, hashing, candidate_vectors, candidate_vectors_gpu, distance_func):
        self._hashing = hashing
        self._candidate_vectors = candidate_vectors
        self._candidate_vectors_gpu = candidate_vectors_gpu
        self._distance_func = distance_func

        self._build_index()

    def _build_index(self):
        indexes = self.hash(self._candidate_vectors_gpu)
        self.index2row = build_index(indexes)

    def hash(self, query_vectors, batch_size=1024):
        hash_keys = []

        n = query_vectors.shape[0]
        n_batches = n // batch_size
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = query_vectors[start:end, :]
            hash_key = self._hashing.hash(batch)
            hash_keys += hash_key
        last_batch = query_vectors[n_batches*batch_size:, :]
        hash_key = self._hashing.hash(last_batch)
        hash_keys += hash_key
        return hash_keys

    def query(self, query_vectors, query_vectors_gpu, k=10) -> List[List[int]]:
        query_indexes = self.hash(query_vectors_gpu)
        recall_result = []
        n_candidates_result = []
        vector_buffer = torch.rand(self._candidate_vectors_gpu.shape).cuda()
        default_empty_rows = torch.LongTensor([]).cuda()
        for idx, qi in enumerate(query_indexes):
            candidate_rows = self.index2row.get(qi, default_empty_rows)

            n_candidates = len(candidate_rows)
            target_vector = query_vectors_gpu[idx, :]

            # NOTE: indexing with tensor will create a copy
            # use index_select will directly move data from one to
            # another. This highly reduce the memory allocation overhead
            torch.index_select(
                self._candidate_vectors_gpu,
                0,
                candidate_rows,
                out=vector_buffer[:n_candidates, :],
            )
            distance = self._distance_func(
                target_vector,
                vector_buffer[:n_candidates, :],
            )
            try:
                topk_idxs = distance.topk(k, largest=False)[1].tolist()
                topk_idxs = [int(candidate_rows[i]) for i in topk_idxs]
            except RuntimeError:
                topk_idxs = candidate_rows
            n_candidates_result.append(n_candidates)
            recall_result.append(topk_idxs)
        return recall_result, n_candidates_result
