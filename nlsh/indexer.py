from typing import List

import torch


def build_index(indexes, cuda=True):
    index2row = {}
    for idx, index_set in enumerate(indexes):
        for index in index_set:
            if index not in index2row:
                index2row[index] = [idx]
            else:
                index2row[index].append(idx)

    # NOTE: this is a import speed optimization
    # allocating a new LongTensor is non trivial and will dominate
    # the evaluation process time
    for index, rows in index2row.items():
        if cuda:
            index2row[index] = torch.LongTensor(rows).cuda()
        else:
            index2row[index] = torch.LongTensor(rows)

    return index2row


class Indexer:

    def __init__(self, hashing, candidate_vectors_gpu, distance_func):
        self._hashing = hashing
        self._candidate_vectors_gpu = candidate_vectors_gpu
        self._distance_func = distance_func

        self._build_index()

    def _build_index(self):
        indexes = self.hash(self._candidate_vectors_gpu, hash_times=1)
        self.index2row = build_index(indexes)

    def hash(self, query_vectors, batch_size=128, hash_times=1):
        hash_keys = []

        n = query_vectors.shape[0]
        n_batches = n // batch_size
        for idx in range(n_batches):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            batch = query_vectors[start:end, :]
            hash_key = self._hashing.hash(batch, hash_times)
            hash_keys += hash_key
        last_batch = query_vectors[n_batches*batch_size:, :]
        hash_key = self._hashing.hash(last_batch)
        hash_keys += hash_key
        return hash_keys

    def query(self, query_vectors, query_vectors_gpu, k=10, hash_times=10) -> List[List[int]]:
        query_indexes = self.hash(query_vectors_gpu, hash_times=hash_times)
        recall_result = []
        n_candidates_result = []
        vector_buffer = torch.rand(self._candidate_vectors_gpu.shape).cuda()
        default_empty_rows = torch.LongTensor([]).cuda()
        for idx, qi in enumerate(query_indexes):
            n_candidates = 0
            target_vector = query_vectors_gpu[idx, :]
            buffer_start = 0
            candidate_rows_list = []
            for key in list(qi):
                candidate_rows = self.index2row.get(key, default_empty_rows)
                candidate_rows_list.append(candidate_rows)
                current_key_candidates = len(candidate_rows)
                n_candidates += current_key_candidates
                buffer_end = buffer_start + current_key_candidates

                # NOTE: indexing with tensor will create a copy
                # use index_select will directly move data from one to
                # another. This highly reduce the memory allocation overhead
                torch.index_select(
                    self._candidate_vectors_gpu,
                    0,
                    candidate_rows,
                    out=vector_buffer[buffer_start:buffer_end, :],
                )
                buffer_start = buffer_end
            distance = self._distance_func(
                target_vector,
                vector_buffer[:buffer_end, :],
            )
            try:
                topk_idxs = distance.topk(k, largest=False)[1].tolist()
                topk_idxs = [int(torch.cat(candidate_rows_list)[i]) for i in topk_idxs]
            except RuntimeError:
                topk_idxs = candidate_rows
            n_candidates_result.append(n_candidates)
            recall_result.append(topk_idxs)
        return recall_result, n_candidates_result
