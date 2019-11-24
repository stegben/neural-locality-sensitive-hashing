from typing import List, Union


def _recall(y_true: List[int], y_pred: List[int]) -> float:
    n_true = len(y_true)
    true_positives = len(set(y_true) & set(y_pred))
    return true_positives / n_true


def calculate_recall(
        y_true: List[List[int]],
        y_pred: List[List[int]],
        reduce_func=None,
    ) -> Union[List[float], float]:
    # TODO: unittest

    assert len(y_true) == len(y_pred)
    recalls = [
        _recall(yt, yp)
        for yt, yp in zip(y_true, y_pred)
    ]

    if reduce_func is not None:
        return reduce_func(recalls)
    return recalls


def n_recall_at_k(y_true, y_pred, n=1, k=10):
    pass
