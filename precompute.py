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


def self_get_knn_numpy_cosine(vectors, k=1000, batch_size=BATCH_SIZE):
    n = vectors.shape[0]
    knn = np.zeros((n, k), dtype=int)
    distance_buffer = np.ones(n, dtype=np.float32)

    norm = np.linalg.norm(vectors, axis=1)
    for idx in tqdm(range(n)):
        target = vectors[idx, :]
        for batch_idx in range((n // batch_size)):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            batch_vector = vectors[start:end, :]
            distance_buffer[start:end] = 1 - np.inner(batch_vector, target) / (norm[idx] * norm[start:end])
        last_batch_vector = vectors[end:, :]
        distance_buffer[end:] = 1 - np.inner(last_batch_vector, target) / (norm[idx] * norm[end:])
        knn[idx, :] = np.argpartition(distance_buffer, k)[:k]
    return knn


# def self_get_knn_tf_cosine(vectors, k=1000, batch_size=64):
#     n = vectors.shape[0]
#     dim = vectors.shape[1]
#     knn = np.zeros((n, k), dtype=int)

#     graph = tf.Graph()
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config, graph=graph)

#     with sess.graph.as_default():
#         target_place = tf.placeholder(tf.float32, [None, dim])
#         candidate_vectors = tf.constant(vectors)
#         inner_products = tf.tensordot(
#             candidate_vectors,
#             tf.transpose(target_place),
#             axes=1,
#         )
#         target_norm = tf.norm(target_place, axis=1)
#         candidate_norm = tf.norm(candidate_vectors, axis=1)
#         norm_mult = tf.tensordot(target_norm, candidate_norm, axes=0)
#         cosine_similarity = tf.transpose(inner_products) / norm_mult
#         cosine_distance = 1 - cosine_similarity
#         top_k = tf.slice(tf.argsort(cosine_distance), begin=[0, 0], size=[-1, k])

#     for batch_idx in tqdm(range((n // batch_size + 1))):
#         start = batch_idx * batch_size
#         end = (batch_idx + 1) * batch_size
#         target = vectors[start:end, :]
#         # distance = sess.run(cosine_distance, {target_place: target})
#         # knn[start:end, :] = np.argpartition(distance, k, axis=-1)[:, :k]
#         knn[start:end, :] = sess.run(top_k, {target_place: target})
#     return knn


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


def self_get_knn_pt_cosine(vectors, k=1000, batch_size=64):
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


def get_train_test_cosine_distance(train, test):
    # TODO: precompute the distance between train and test can reduce the evaluation time
    pass


if __name__ == "__main__":
    data_path_key = sys.argv[1]
    data_path = DATA_PATHS[data_path_key]
    f_data = h5py.File(data_path, 'r')
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    # train_knn = self_get_knn_tf_cosine(train_data)
    train_knn = self_get_knn_pt_cosine(train_data)
    ground_truth = np.array(f_data['neighbors'])
    distances = np.array(f_data['distances'])

    # train_test_distance =

    preprocessed_path = data_path+'.processed'
    f_preprocessed_data = h5py.File(preprocessed_path, 'w')
    f_preprocessed_data.create_dataset("train", data=train_data)
    f_preprocessed_data.create_dataset("train_knn", data=train_knn)
    f_preprocessed_data.create_dataset("test", data=test_data)
    f_preprocessed_data.create_dataset("neighbors", data=ground_truth)
    f_preprocessed_data.create_dataset("distances", data=distances)

    f_preprocessed_data.close()
