from time import time

import numpy as np
from tqdm import tqdm

from nlsh.metrics import calculate_recall
import hnswlib


class HierarchicalNavigableSmallWorldGraph:

    def __init__(self, data, logger, *args, **kwargs):

        self._data = data
        self._logger = logger

        self._max_connections = 10
        self._ef_construction = 500

        if not self._data.prepared:
            self._data.load()

        self.candidate_vectors = self._data.training
        self.validation_data = self._data.testing
        self.candidate_self_knn = self._data.training_self_knn
        self.ground_truth = self._data.ground_truth[:, :10]

        self.index = hnswlib.Index(space='cosine', dim=self.candidate_vectors.shape[1])
        self.index.init_index(
            max_elements=self.candidate_vectors.shape[0],
            M=self._max_connections,
            ef_construction=self._ef_construction,
        )
        self.index.set_ef(40)

    def fit(self, K, batch_size=4096, *args, **kwargs):
        n = self.candidate_vectors.shape[0]
        n_batch = n // batch_size
        idxs = np.arange(n)
        np.random.shuffle(idxs)

        for batch_idx in tqdm(range(n_batch)):
            start = batch_idx * batch_size
            end = start + batch_size
            selected_idxs = idxs[start:end]
            self.index.add_items(self.candidate_vectors[selected_idxs, :], selected_idxs)
        selected_idxs = idxs[end:]
        self.index.add_items(self.candidate_vectors[selected_idxs, :], selected_idxs)

        # Validation
        t1 = time()
        predict_knns, _, counts = self.index.knn_query(self.validation_data, k=K)
        t2 = time()
        query_time = t2 - t1
        current_recall = calculate_recall(self.ground_truth, predict_knns, np.mean)
        current_query_size = np.mean(counts)
        global_step = 1
        self._logger.log("test/recall", current_recall, global_step)
        print(current_recall)
        self._logger.log("test/query_size", current_query_size, global_step)
        print(current_query_size)
        qps = self.validation_data.shape[0] / query_time
        self._logger.log("test/qps", qps, global_step)
