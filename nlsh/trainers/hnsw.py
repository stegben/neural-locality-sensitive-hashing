from time import time
import math
from queue import PriorityQueue

import numpy as np
import networkx as nx

from nlsh.metrics import calculate_recall


class HierarchicalNavigableSmallWorldGraph:

    def __init__(self, data, *args, **kwargs):
        self._data = data
        self.graphs = []
        self.default_enter_point = -1

        self._max_layers = 10
        for _ in range(self._max_layers):
            self.graphs.append(nx.Graph())
        self._max_connections = 100
        self._max_connections_per_layer = 20
        self._ef_construction = 100

        if not self._data.prepared:
            self._data.load()

        self.candidate_vectors = self._data.training
        self.validation_data = self._data.testing
        self.candidate_self_knn = self._data.training_self_knn
        self.ground_truth = self._data.ground_truth[:, :K]

    def insert(self, idx, value):
        l = math.floor(-math.log(random.random()) * self._max_layers)
        starting_layer = l if l < self._max_layers else (self._max_layers - 1)
        enter_node = self.default_enter_point

        if self.default_enter_point == -1:  # inserting the first point

            self.default_enter_point = idx
            return

        # get enter point from the starting layer
        for graph in reversed(self.graphs[starting_layer: self._max_layers]):
            enter_node, _ = self.search_nn(graph, enter_node, q)

        # start connecting each layers
        for graph in reversed(self.graphs[:self.starting_layer]):
            results, _ = self.search_layer(graph, )
            to_be_connected = [results.get()[1] for _ in range(self._max_connections)]
            graph.add_node(idx)
            for node in to_be_connected:
                graph.add_edge(idx, node)

            # After adding the edges, existed nodes may have edges exceed max_connections_per_layer, so shrink them.

        enter_node = self.default_enter_point
        for graph in reversed(self.graphs[1:]):
            enter_node, _ = self.search_nn(graph, enter_node, q)

    def search_nn(
            self,
            nswg: nx.Graph,
            enter_node: int,
            q: np.ndarray,
        ):
        cur_node = enter_node
        smallest_dist = self._data.distance(q, self.candidate_vectors[enter_node, :])
        count = 1
        changed = True
        while changed:
            changed = False

            for node in nswg.neighbors(cur_node):
                cur_dist = self._data.distance(q, self.candidate_vectors[enter_node, :])
                if cur_dist < smallest_dist:
                    changed = True
                    smallest_dist = cur_dist
                    cur_node = node
                count += 1
        return cur_node, count

    def search_layer(
            self,
            nswg: nx.Graph,
            enter_node: int,
            q: np.ndarray,
            ef: int,
        ):
        enter_distance = self._data.distance(q, self.candidate_vectors[enter_node, :])[0]
        calculated = 1
        candidates = PriorityQueue()
        candidates.put((enter_distance, enter_node))
        results = PriorityQueue()
        results.put((-enter_distance, enter_node))
        visited = set([enter_node])

        while not candidates.empty():
            cand_smallest_dist, cand_best_node = candidates.get(0)
            cur_largest_dist, _ = results.queue[0]
            cur_largest_dist = -cur_largest_dist

            if cand_smallest_dist > cur_largest_dist:
                break

            for neighbor in nswg.neighbor(cand_best_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    cur_dist = self._data.distance(q, self.candidate_vectors[neighbor, :])[0]
                    calculated += 1
                    cur_largest_dist, _ = results.queue[0]
                    cur_largest_dist = -cur_largest_dist

                    if (len(results.queue) < ef) or (cur_dist < cur_largest_dist):
                        candidates.put((cur_dist, neighbor))
                        results.put((-cur_dist, neighbor))

                    if len(results.queue) > ef:
                        results.get()  # pop head
        return results, calculated

    def search_knn(self, q: np.ndarray, k: int):
        # enter_node = random.randint(low=0, high=self.candidate_vectors.shape[0])
        enter_node = self.default_enter_point
        count = 0
        for graph in reversed(self.graphs[1:]):
            enter_node, cur_count = self.search_nn(graph, enter_node, q)
            count += cur_count
        results = self.search_layer(self.graphs[0], enter_node, q, ef=10)
        knn = [results.get()[1] for _ in range(k)]
        return knn, counts

    def fit(self, K, batch_size=1024, learning_rate=3e-4, test_every_updates=1000):
        # Validation
        t1 = time()
        recalls, n_candidates = indexer.query(self._validation_data_gpu, k=K)
        t2 = time()
        query_time = t2 - t1
        current_recall = calculate_recall(list(ground_truth), recalls, np.mean)
        current_query_size = np.mean(n_candidates)

        if (current_recall > best_recall) and (current_query_size < best_query_size):
            base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
            self._hashing.save(base_name)
            best_recall = current_recall

        self._logger.log("test/recall", current_recall, global_step)
        self._logger.log("test/query_size", current_query_size, global_step)
        qps = self._validation_data.shape[0] / query_time
        self._logger.log("test/qps", qps, global_step)

        # Evaluate training set (see if overfit)
        recalls, n_candidates = indexer.query(sampled_train, k=K)
        train_recall = calculate_recall(list(sampled_train_ground_truth), recalls, np.mean)
        train_query_size = np.mean(n_candidates)
        self._logger.log("training/recall", train_recall, global_step)
        self._logger.log("training/query_size", train_query_size, global_step)