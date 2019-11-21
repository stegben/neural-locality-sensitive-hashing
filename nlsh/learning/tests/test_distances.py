import numpy as np
import torch

from ..distances import JSD_categorical, KL_multivariate_bernoulli


def test_JSD_categorical():
    p = torch.Tensor([[0.1, 0.9], [0.1, 0.9], [1., 0.]])
    q = torch.Tensor([[0.5, 0.5], [0.1, 0.9], [0., 1.]])
    divergence = JSD_categorical(p, q)
    np.testing.assert_array_almost_equal(
        divergence.numpy(),
        np.array([0.101749, 0., 0.693147]),
    )


def test_KL_multivariate_bernoulli():
    p = torch.Tensor([[0.5, 0.5], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [1., 0.]])
    q = torch.Tensor([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.1, 0.9], [0., 1.]])
    divergence = KL_multivariate_bernoulli(p, q)
    np.testing.assert_array_almost_equal(
        divergence.numpy(),
        np.array([0.510826, 0.368064, 1.757779, 0., np.nan]),
    )
