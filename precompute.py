import os
import sys

from dotenv import load_dotenv
import h5py
import numpy as np
from tqdm import tqdm

import tensorflow as tf

load_dotenv()
DATA_PATHS = {
    "glove_25": os.environ.get("NLSH_GLOVE_25_PATH"),
    "glove_50": os.environ.get("NLSH_GLOVE_50_PATH"),
    "glove_100": os.environ.get("NLSH_GLOVE_100_PATH"),
    "glove_200": os.environ.get("NLSH_GLOVE_200_PATH"),
    # TODO: sift
}
BATCH_SIZE = 2048 * 128


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


def self_get_knn_tf_cosine(vectors, k=1000, batch_size=BATCH_SIZE):
    n = vectors.shape[0]
    dim = vectors.shape[1]
    knn = np.zeros((n, k), dtype=int)
    distance_buffer = np.ones(n, dtype=np.float32)
    norm = np.linalg.norm(vectors, axis=1)

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=graph)

    def vector_gen():
        # for batch_idx in range((n // batch_size + 1)):
        #     start = batch_idx * batch_size
        #     end = (batch_idx + 1) * batch_size
        #     batch_vector = vectors[start:end, :]
        #     yield batch_vector
        for idx in range(n):
            return vectors[idx, :]

    with sess.graph.as_default():
        target_place = tf.placeholder(tf.float32, [dim])
        dataset = tf.data.Dataset.from_generator(
            vector_gen,
            tf.float32,
            tf.TensorShape([dim]),
        ).batch(batch_size).prefetch(1)
        iterator = dataset.make_initializable_iterator()
        candidate_vectors = iterator.get_next()
        cosine_distance = tf.losses.cosine_distance(
            tf.transpose(target_place),
            candidate_vectors,
            axis=-1,
            reduction=tf.losses.Reduction.NONE,
        )


    for idx in tqdm(range(n)):
        target = vectors[idx, :]
        sess.run(iterator.initializer)
        start = 0
        while True:
            try:
                distance = sess.run(cosine_distance, {target_place: target})
                end = start + distance.shape[0]
                distance_buffer[start:end] = distance
                start = end
            except tf.errors.OutOfRangeError:
                break
        knn[idx, :] = np.argpartition(distance_buffer, k)[:k]
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
    train_knn = self_get_knn_tf_cosine(train_data)
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
