import os
import sys

from dotenv import load_dotenv
import h5py
import numpy as np
from tqdm import tqdm

import torch

load_dotenv()
DATA_PATHS = {
    "glove_25": os.environ.get("NLSH_GLOVE_25_PATH"),
    "glove_50": os.environ.get("NLSH_GLOVE_50_PATH"),
    "glove_100": os.environ.get("NLSH_GLOVE_100_PATH"),
    "glove_200": os.environ.get("NLSH_GLOVE_200_PATH"),
    # TODO: sift
}
BATCH_SIZE = 2048 * 512


def _cosine_distance(v1, v2):
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


def self_get_knn_pt_cosine(vectors, k=100, batch_size=512):
    vector_pt = torch.from_numpy(vectors).cuda()
    n = vectors.shape[0]
    knn = np.zeros((n, k), dtype=int)

    for batch_idx in tqdm(range((n // batch_size + 1))):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        target = vector_pt[start:end, :]
        knn[start:end, :] = _cosine_distance(target, vector_pt).topk(k+1, dim=1, largest=False)[1][:, 1:].cpu().numpy()
    return knn


def self_get_knn_numpy_l2(vectors, k=1000, batch_size=BATCH_SIZE):
    # TODO: for SIFT, MNIST, GIST, Fashion-MNIST
    pass


if __name__ == "__main__":
    data_path_key = sys.argv[1]
    data_path = DATA_PATHS[data_path_key]
    f_data = h5py.File(data_path, 'r')
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])
    distances = np.array(f_data['distances'])

    train_knn = self_get_knn_pt_cosine(train_data)

    preprocessed_path = data_path+'.processed'
    f_preprocessed_data = h5py.File(preprocessed_path, 'w')
    f_preprocessed_data.create_dataset("train", data=train_data)
    f_preprocessed_data.create_dataset("train_knn", data=train_knn)
    f_preprocessed_data.create_dataset("test", data=test_data)
    f_preprocessed_data.create_dataset("neighbors", data=ground_truth)
    f_preprocessed_data.create_dataset("distances", data=distances)

    f_preprocessed_data.close()
