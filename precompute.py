import os
import cProfile as Profile

from dotenv import load_dotenv
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


load_dotenv()
DATA_PATH = os.environ.get("NLSH_GLOVE_25_PATH")
BATCH_SIZE = 2048 * 16


class PureDataset(Dataset):

    def __init__(self, vectors):
        self._vectors = vectors

    def __getitem__(self, idx):
        return self._vectors[idx, :]

    def __len__(self):
        return self._vectors.shape[0]


def cosine_distance(v1, v2, dim=-1):
    return 1 - F.cosine_similarity(v1, v2, dim=dim)


def self_get_knn_torch(vectors, distance_func, k=1000):
    n = vectors.shape[0]
    knn = np.zeros((n, k), dtype=int)
    distance_buffer = np.ones(n, dtype=np.float32)
    dataset = PureDataset(vectors)
    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for idx in tqdm(range(n)):
        if idx > 10:
            break
        target = vectors[idx, :].cuda()
        for batch_idx, batch_vector in enumerate(dl):
            batch_vector = batch_vector.cuda()
            start = batch_idx * BATCH_SIZE
            end = (batch_idx + 1) * BATCH_SIZE
            distance_buffer[start:end] = distance_func(target, batch_vector).cpu()
        knn[idx, :] = distance_buffer.argsort()[:k]
    return knn


def self_get_knn_numpy_cosine(vectors, distance_func, k=1000):
    n = vectors.shape[0]
    knn = np.zeros((n, k), dtype=int)
    distance_buffer = np.ones(n, dtype=np.float32)

    norm = np.linalg.norm(vectors, axis=1)
    for idx in tqdm(range(n)):
        target = vectors[idx, :]
        for batch_idx in range(n // BATCH_SIZE):
            start = batch_idx * BATCH_SIZE
            end = (batch_idx + 1) * BATCH_SIZE
            batch_vector = vectors[start:end, :]
            distance_buffer[start:end] = 1 - np.inner(batch_vector, target) / (norm[idx] * norm[start:end])
        knn[idx, :] = np.argpartition(distance_buffer, k)[:k]
    return knn


if __name__ == "__main__":
    f_data = h5py.File(DATA_PATH)
    train_data = torch.from_numpy(np.array(f_data['train']))
    test_data = np.array(f_data['test'])

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    train_knn = self_get_knn_numpy_cosine(train_data, cosine_distance)
    pr.disable()
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr).sort_stats(sortby).reverse_order()
    ps.print_stats()
    print(s.getvalue())
    import ipdb; ipdb.set_trace()
