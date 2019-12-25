import torch

from ..indexer import build_index


def test_build_index():
    indexes = [
        set([1, 2]),
        set([2, 3, 4]),
        set([1, 5]),
    ]
    index2row = build_index(indexes, cuda=False)
    expected = {
        1: torch.LongTensor([0, 2]),
        2: torch.LongTensor([0, 1]),
        3: torch.LongTensor([1]),
        4: torch.LongTensor([1]),
        5: torch.LongTensor([2]),
    }
    assert index2row.keys() == expected.keys()

    for k in index2row.keys():
        expected_idxs = expected[k]
        actual_idxs = index2row[k]

        assert torch.equal(expected_idxs, actual_idxs)
