# distance function for hash code
import abc

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


L2 = F.pairwise_distance
L2_categorical = F.pairwise_distance


class _Distance(abc.ABC):

    @abc.abstractmethod
    def rowwise(self, p, q):
        """
        p: (n, k)
        q: (n, k)

        outputs d: (n)
        """
        pass

    @abc.abstractmethod
    def pairwise(self, p, q):
        """
        p: (n, k)
        q: (m, k)

        outputs d: (n, m)
        """
        pass

    @abc.abstractmethod
    def row_pairwise(self, p, q):
        """
        p: (n, m, k)
        q: (n, p, k)

        outputs d: (n, m, p)
        """
        pass


def JSD_categorical(p, q):
    """
    p: (n, k)
    q: (n, k)

    outputs d: (n)
    """
    m = (p + q) / 2

    p = Categorical(probs=p)
    q = Categorical(probs=q)
    m = Categorical(probs=m)

    kl_pm = kl_divergence(p, m)
    kl_qm = kl_divergence(q, m)
    return (kl_pm + kl_qm) / 2


def hellinger_categorical(p, q):
    """hellinger divergence between categorical distributions
    p: (n, k)
    q: (n, k)

    outputs d: (n)
    """
    p_sqrt = torch.sqrt(p)
    q_sqrt = torch.sqrt(q)
    return F.pariwise_distance(p_sqrt, q_sqrt) / torch.sqrt(2)


def KL_multivariate_bernoulli(p, q, epsilon=1e-16):
    """KL divergence between 2 multivariate bernoulli
    p: (n, k)
    q: (n, k)

    outputs d: (n)
    """
    positive = p * torch.log(epsilon + p / (q + 1e-20))
    negative = (1 - p) * torch.log(epsilon + (1 - p) / (1 - q + 1e-20))
    return torch.mean(positive + negative, 1)


def _pairwise_KL_multivariate_bernoulli(p, q, epsilon=1e-16):
    """KL divergence between 2 multivariate bernoulli
    p: (n, k)
    q: (m, k)

    outputs d: (n, m)
    """
    log_p_q = torch.log(epsilon + torch.einsum('nk,mk->nmk', p, 1 / (q + 1e-20)))
    p_log_p_q = p[:, None, :] * log_p_q
    positive = p_log_p_q.sum(-1)

    log_np_nq = torch.log(epsilon + torch.einsum('nk,mk->nmk', 1-p, 1 / (1-q + 1e-20)))
    np_log_np_nq = (1 - p[:, None, :]) * log_np_nq
    negative = np_log_np_nq.sum(-1)
    return positive + negative


def _row_pairwise_KL_multivariate_bernoulli(p, q, epsilon=1e-16):
    """KL divergence between 2 multivariate bernoulli
    p: (n, m, k)
    q: (n, p, k)

    outputs d: (n, m, p)
    """
    log_p_q = torch.log(epsilon + torch.einsum('nmk,npk->nmpk', p, 1 / (q + 1e-20)))
    p_log_p_q = p[:, :, None, :] * log_p_q
    positive = p_log_p_q.sum(-1)

    log_np_nq = torch.log(epsilon + torch.einsum('nmk,npk->nmpk', 1-p, 1 / (1-q + 1e-20)))
    np_log_np_nq = (1 - p[:, :, None, :]) * log_np_nq
    negative = np_log_np_nq.sum(-1)
    return positive + negative


def _entropy_multivariate_bernoulli(p, epsilon):
    positive = - p * torch.log(p + epsilon)
    negative = - (1 - p) * torch.log(1 - p + epsilon)
    return torch.mean(positive + negative, -1)


def cross_entropy_multivariate_bernoulli(p, q, epsilon=1e-20):
    kl = KL_multivariate_bernoulli(p, q, epsilon)
    entropy = _entropy_multivariate_bernoulli(p, epsilon)
    return  kl + entropy


class MVBernoulliKLDivergence(_Distance):

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def rowwise(self, p, q):
        """
        p: (n, k)
        q: (n, k)

        outputs d: (n)
        """
        return KL_multivariate_bernoulli(p, q, self._epsilon)

    def pairwise(self, p, q):
        """
        p: (n, k)
        q: (m, k)

        outputs d: (n, m)
        """
        return _pairwise_KL_multivariate_bernoulli(p, q, self._epsilon)

    def row_pairwise(self, p, q):
        """
        p: (n, m, k)
        q: (n, p, k)

        outputs d: (n, m, p)
        """
        return _row_pairwise_KL_multivariate_bernoulli(p, q, self._epsilon)


class MVBernoulliCrossEntropy(_Distance):

    def __init__(self, epsilon):
        self._epsilon = epsilon

    def rowwise(self, p, q):
        """
        p: (n, k)
        q: (n, k)

        outputs d: (n)
        """
        kl = KL_multivariate_bernoulli(p, q, self._epsilon)
        entropy = _entropy_multivariate_bernoulli(p, self._epsilon)
        return  kl + entropy

    def pairwise(self, p, q):
        """
        p: (n, k)
        q: (m, k)

        outputs d: (n, m)
        """
        pairwise_kl = _pairwise_KL_multivariate_bernoulli(p, q, self._epsilon)
        entropy = _entropy_multivariate_bernoulli(p, self._epsilon)
        return  pairwise_kl + entropy[:, None]

    def row_pairwise(self, p, q):
        """
        p: (n, m, k)
        q: (n, p, k)

        outputs d: (n, m, p)
        """
        row_pairwise_kl = _row_pairwise_KL_multivariate_bernoulli(p, q, self._epsilon)
        entropy = _entropy_multivariate_bernoulli(p, self._epsilon)
        return  row_pairwise_kl + entropy[:, :, None]


class MVBernoulliL2(_Distance):

    def rowwise(self, p, q):
        """
        p: (n, k)
        q: (n, k)

        outputs d: (n)
        """
        return F.pairwise_distance(p, q)

    def pairwise(self, p, q):
        """
        p: (n, k)
        q: (m, k)

        outputs d: (n, m)
        """
        return torch.cdist(p, q, p=2)

    def row_pairwise(self, p, q):
        """
        p: (n, m, k)
        q: (n, p, k)

        outputs d: (n, m, p)
        """
        return torch.cdist(p, q, p=2)
