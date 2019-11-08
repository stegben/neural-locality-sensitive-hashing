# distance function for hash code
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


L2 = F.pairwise_distance
L2_categorical = F.pairwise_distance


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
    """Earth mover's distance between categorical distributions
    p: (n, k)
    q: (n, k)

    outputs d: (n)
    """
    p_sqrt = torch.sqrt(p)
    q_sqrt = torch.sqrt(q)
    return F.pariwise_distance(p_sqrt, q_sqrt) / torch.sqrt(2)


def L2_multivariate_bernoulli(p1, p2):
    pass
