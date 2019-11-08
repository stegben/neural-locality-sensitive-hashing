import os
import argparse
from typing import List, Set

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from nlsh.data import Glove
from nlsh.trainers import build_index, calculate_recall

load_dotenv()

MODEL_SAVE_DIR = os.environ["NLSH_MODEL_SAVE_DIR"]


def nlsh_eval_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--data_id",
        type=str,
        choices=("glove_25", "glove_50", "glove_100", "glove_200",),
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
    )
    return parser


def get_data_by_id(data_id):
    id2path = {
        "glove_25": os.environ.get("NLSH_PROCESSED_GLOVE_25_PATH"),
        "glove_50": os.environ.get("NLSH_PROCESSED_GLOVE_50_PATH"),
        "glove_100": os.environ.get("NLSH_PROCESSED_GLOVE_100_PATH"),
        "glove_200": os.environ.get("NLSH_PROCESSED_GLOVE_200_PATH"),
    }
    return Glove(id2path[data_id])


def _binarr_to_int(binarr):
    out = 0
    for bit in binarr:
        out = (out << 1) | bit
    return out


def hash_all(hasher, query_vectors):
    probs = hasher(query_vectors)
    codes = (probs > 0.5).tolist()
    hash_results = []
    for binarr in codes:
        hash_results.append(_binarr_to_int(binarr))
    return hash_results


def sample_and_collect(dist, n) -> List[Set]:
    codes = dist.sample((n,)).int().permute(1, 0, 2).tolist()  # (batch_size, n, code_size)
    hash_results = []
    for binarrs in codes:
        hashes = []
        for binarr in binarrs:
            hashes.append(_binarr_to_int(binarr))
        hash_results.append(set(hashes))
    return hash_results


def hash_by_batch(hasher, query_vectors, batch_size):
    hash_keys = []

    n = query_vectors.shape[0]
    n_batches = n // batch_size
    for idx in range(n_batches):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        batch = query_vectors[start:end, :]
        hash_key = hash_all(hasher, batch)
        hash_keys += hash_key
    last_batch = query_vectors[n_batches*batch_size:, :]
    hash_key = hash_all(hasher, last_batch)
    hash_keys += hash_key
    return hash_keys


def main():
    parser = nlsh_eval_argparse()
    args = parser.parse_args()

    model_path = args.model_path
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_SAVE_DIR, model_path)
    data_id = args.data_id
    K = args.k

    hasher = torch.jit.load(model_path)
    data = get_data_by_id(data_id)
    data.load()

    candidate_vectors = torch.from_numpy(data.training)
    indexes = hash_by_batch(hasher, candidate_vectors, 4096)
    index2row = build_index(indexes)
    index2rownum = {
        k: v.shape[0]
        for k, v in index2row.items()
    }
    # hash eval
    query_vectors = torch.from_numpy(data.testing)
    test_probs = hasher(query_vectors)
    test_dist = torch.distributions.Bernoulli(test_probs)

    ground_truth = data.ground_truth[:, :K]

    for n_samples in range(1, 11):
        test_indexes = sample_and_collect(test_dist, n_samples)

        test_indexes_flattened = [qi for qis in test_indexes for qi in list(qis)]
        test_candidate_nums = [index2rownum.get(qi, 0) for qi in test_indexes_flattened]
        import ipdb; ipdb.set_trace()
        result = []
        list_n_candidates = []
        vector_buffer = torch.rand(candidate_vectors.shape)
        idx_buffer = torch.LongTensor([i for i in range(candidate_vectors.shape[0])])
        for idx, qis in enumerate(tqdm(test_indexes)):
            target_vector = query_vectors[idx, :]
            start = 0
            for qi in list(qis):
                candidate_rows = index2row.get(qi, torch.LongTensor([]))
                n_candidates = len(candidate_rows)

                # NOTE: indexing with tensor will create a copy
                # use index_select will directly move data from one to
                # another. This highly reduce the memory allocation overhead
                torch.index_select(
                    candidate_vectors,
                    0,
                    candidate_rows,
                    out=vector_buffer[start:start + n_candidates, :],
                )
                idx_buffer[start:start + n_candidates] = candidate_rows
                start += n_candidates
            total_candidates = start
            list_n_candidates.append(total_candidates)
            distance = data.distance(
                target_vector,
                vector_buffer[:total_candidates, :],
            )
            try:
                topk_idxs = distance.topk(K, largest=False)[1].tolist()
                topk_idxs = [int(idx_buffer[i]) for i in topk_idxs]
            except RuntimeError:
                topk_idxs = idx_buffer[:total_candidates]

            result.append(topk_idxs)

        recall = np.mean([
            calculate_recall(y_true, y_pred)
            for y_pred, y_true in zip(result, list(ground_truth))
        ])
        avg_n_candidates = np.mean(list_n_candidates)

        print(avg_n_candidates, recall)
    import ipdb; ipdb.set_trace()
    #   calculate n_candidates
    #   calculate recall


if __name__ == "__main__":
    main()