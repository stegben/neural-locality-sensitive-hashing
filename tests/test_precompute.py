import numpy as np

from precompute import self_get_knn_numpy_cosine


def test_self_get_knn_numpy_cosine():
    vectors = np.array([
        [1, 2, 3],
        [3, 2, 1],
        [1, 2, 4],
        [6, 4, 2],
        [2, 4, 6],
    ])
    result = self_get_knn_numpy_cosine(vectors, k=2, batch_size=2)
    assert [set(r) for r in list(result)] == [
        set([0, 4]),
        set([1, 3]),
        set([2, 0]),
        set([3, 1]),
        set([4, 0]),
    ]
