import numpy as np
import torch

from ..distances import JSD_categorical


def test_JSD_categorical():
    p = torch.Tensor([[0.1, 0.9], [0.1, 0.9], [1., 0.]])
    q = torch.Tensor([[0.5, 0.5], [0.1, 0.9], [0., 1.]])
    divergence = JSD_categorical(p, q)
    np.testing.assert_array_almost_equal(
        divergence.numpy(),
        np.array([0.101749, 0., 0.693147]),
    )
