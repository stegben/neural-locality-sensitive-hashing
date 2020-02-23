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
    "sift": os.environ.get("NLSH_SIFT_PATH"),
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


def _l2(v1, v2):
    """l2 distance betwenn 2 matrix

    v1: (n, d)
    v2: (m, d)

    Returns
    D: (n, m) where D[i, j] is the distance between v1[i, :] and v2[j, :]
    """
    v1_norm = v1.pow(2).sum(dim=-1, keepdim=True)
    v2_norm = v2.pow(2).sum(dim=-1, keepdim=True)
    result = torch.addmm(
        v2_norm.transpose(-2, -1),
        v1,
        v2.transpose(-2, -1),
        alpha=-2,
    ).add_(v1_norm)
    return result


def self_get_knn_pt(vectors, distance_func, k=100, batch_size=512):
    vector_pt = torch.from_numpy(vectors).cuda()
    n = vectors.shape[0]
    knn = np.zeros((n, k), dtype=int)

    for batch_idx in tqdm(range((n // batch_size + 1))):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size
        target = vector_pt[start:end, :]
        knn[start:end, :] = distance_func(target, vector_pt).topk(k+1, dim=1, largest=False)[1][:, 1:].cpu().numpy()
    return knn


DISTANCE_FUNC = {
    "glove_25": _cosine_distance,
    "glove_50": _cosine_distance,
    "glove_100": _cosine_distance,
    "glove_200": _cosine_distance,
    "sift": _l2,
}


if __name__ == "__main__":
    data_path_key = sys.argv[1]
    data_path = DATA_PATHS[data_path_key]
    f_data = h5py.File(data_path, 'r')
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])
    distances = np.array(f_data['distances'])

    distance_func = DISTANCE_FUNC[data_path_key]
    train_knn = self_get_knn_pt(train_data, distance_func)

    preprocessed_path = data_path+'.processed'
    f_preprocessed_data = h5py.File(preprocessed_path, 'w')
    f_preprocessed_data.create_dataset("train", data=train_data)
    f_preprocessed_data.create_dataset("train_knn", data=train_knn)
    f_preprocessed_data.create_dataset("test", data=test_data)
    f_preprocessed_data.create_dataset("neighbors", data=ground_truth)
    f_preprocessed_data.create_dataset("distances", data=distances)

    f_preprocessed_data.close()
    f_data.close()
