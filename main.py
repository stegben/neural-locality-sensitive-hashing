import h5py
import numpy as np
from scipy.spatial.distance import pdist

DATA_PATH = "../data/glove-25-angular.hdf5"
K = 10


class _Index:

    def __init__(self, candidate_vectors, index2row, distance_func):
        self._candidate_vectors = candidate_vectors
        self._index2row = index2row
        self._distance_func = distance_func

    def query(self, indexes):



class NeuralLocalitySensitiveHashing:

    def __init__(self, encoder, learner, distance_func):
        self._encoder = encoder
        self._learner = learner
        self._distance_func = distance_func

    def fit(self, candidate_vectors):
        self.encoder()
        return index

    def hash(self, query_vectors):
        pass


def main():
    f_data = h5py.File(DATA_PATH)
    train_data = np.array(f_data['train'])
    test_data = np.array(f_data['test'])
    ground_truth = np.array(f_data['neighbors'])

    nlsh = ...()

    indexes = nlsh.fit(train_data)
    hashed_data = nlsh.hash(test_data)

    indexes.query(hashed_data)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
