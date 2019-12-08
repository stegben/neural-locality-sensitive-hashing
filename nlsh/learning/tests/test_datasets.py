import pytest
import torch

from ..datasets import KNearestNeighborAllOut


def test_knn_all_out():
    vectors = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [1, 3, 5],
        [2, 4, 6],
    ], dtype=torch.float32)
    knn = torch.tensor([
        [1, 3, 4],
        [4, 2, 3],
        [3, 0, 1],
        [2, 4, 1],
        [0, 2, 1],
    ], dtype=torch.long)
    k = 2

    ds = KNearestNeighborAllOut(
        candidate_vectors=vectors,
        candidate_self_knn=knn,
        k=k,
    )

    gen = ds.batch_generator(batch_size=2, shuffle=False)

    anchor, positives = next(gen)
    assert torch.allclose(anchor, torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ], dtype=torch.float32))
    assert torch.allclose(positives, torch.tensor([
        [[4, 5, 6], [1, 3, 5]],
        [[2, 4, 6], [7, 8, 9]],
    ], dtype=torch.float32))

    anchor, positives = next(gen)
    assert torch.allclose(anchor, torch.tensor([
        [7, 8, 9],
        [1, 3, 5],
    ], dtype=torch.float32))
    assert torch.allclose(positives, torch.tensor([
        [[1, 3, 5], [1, 2, 3]],
        [[7, 8, 9], [2, 4, 6]],
    ], dtype=torch.float32))

    with pytest.raises(StopIteration):
        next(gen)
