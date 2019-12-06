import numpy as np

from precompute import self_get_knn_pt_cosine


def test_self_get_knn_pt_cosine():
    vectors = np.array([
        [1.2, 2, 3],
        [3, 2, 1],
        [1, 2, 4],
        [6, 4, 2.5],
        [2, 4, 6],
    ], dtype=np.float32)
    result = self_get_knn_pt_cosine(vectors, k=2, batch_size=2)
    assert [set(r) for r in list(result)] == [
        set([4, 2]),
        set([3, 0]),
        set([0, 4]),
        set([1, 0]),
        set([0, 2]),
    ]
