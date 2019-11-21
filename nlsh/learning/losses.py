import torch
import torch.nn.functional as F


def contrastive_loss(
        anchor,
        other,
        label,
        distance_func,
        negative_margin=0.1,
        positive_margin=0.0,
    ):
    d = distance_func(anchor, other)
    positive_loss = label * torch.clamp(d - positive_margin, min=0)**2
    negative_loss = (1 - label) * torch.clamp(d - negative_margin, max=0)**2
    return torch.mean(positive_loss + negative_loss) / 2


def triplet_loss(
        anchor,  # (n, d)
        pos,  # (n, d)
        neg,  # (n, d)
        distance_func,
        margin=0.1,
    ):
    d_pos = distance_func(anchor, pos)  # (n)
    d_neg = distance_func(anchor, neg)  # (n)
    loss_vec = torch.clamp(d_pos - d_neg + margin, min=0)
    return torch.mean(loss_vec)
