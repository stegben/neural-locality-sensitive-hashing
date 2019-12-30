from time import time

import torch
import torch.nn.functional as F
import numpy as np

from .base import Trainer


class OnePass:

    def __init__(self, candidate_vectors):
        self._candidate_vectors = candidate_vectors
        self.n_candidates = self._candidate_vectors.shape[0]

    def __len__(self):
        return self.n_candidates

    def batch_generator(self, batch_size, shuffle=False):
        idxs = np.arange(len(self))
        if shuffle:
            np.random.shuffle(idxs)

        n_batches = len(self) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            vector = self._candidate_vectors[idxs[start:end], :]
            yield vector


class StraightThroughCodebookLookup(torch.autograd.Function):
    """Straight Through gradient ArgMax

    ref: https://discuss.pytorch.org/t/differentiable-argmax/33020/4
    """
    @staticmethod
    def forward(ctx, inputs, codebook_weight):
        """
        inputs: probability of codebooks, shape is (batch_size, n_codes)
        codebook_weight: weight of codebook, shape is (n_codes, data_dim)
        """
        idx = torch.argmax(inputs, dim=1)
        ctx._inputs_shape = inputs.shape
        ctx._inputs_dtype = inputs.dtype
        ctx._inputs_device = inputs.device
        ctx.mark_non_differentiable(idx)
        ctx.save_for_backward(idx, codebook_weight)
        return codebook_weight.index_select(dim=0, index=idx)

    @staticmethod
    def backward(ctx, grad_output):
        idx, codebook_weight = ctx.saved_tensors

        grad_inputs, grad_codebook_weight = None, None

        # if ctx.needs_input_grad[0]:
        grad_inputs = torch.zeros(ctx._inputs_shape, device=ctx._inputs_device, dtype=ctx._inputs_dtype)
        grad_inputs.scatter_(1, idx[:, None], grad_output.norm(dim=1, keepdim=True))

        # if ctx.needs_input_grad[1]:
        embedding_size = codebook_weight.shape[1]

        grad_output_flatten = (grad_output.contiguous()
                                            .view(-1, embedding_size))
        grad_codebook_weight = torch.zeros_like(codebook_weight)
        grad_codebook_weight.index_add_(0, idx, grad_output_flatten)

        return grad_inputs, grad_codebook_weight


class VQVAE(Trainer):

    def __init__(
            self,
            hashing,
            data,
            model_save_dir,
            logger,
        ):
        self._hashing = hashing
        self._data = data
        self._model_save_dir = model_save_dir
        self._logger = logger

    def _get_dataset(self, vectors, self_knn):
        """
        vectors: torch float tensor, (n, d)
        knn: numpy long array, (n, k), precalculated knn of `vectors`
        """
        dataset = OnePass(vectors)
        return dataset

    def _get_loss(self, batch):
        probs = self._hashing.predict(batch)
        probs.retain_grad()
        codes = StraightThroughCodebookLookup.apply(probs, self.codebook.weight)
        codes.retain_grad()
        dist = F.pairwise_distance(codes, batch)
        loss = (dist**2).mean()
        return loss

    def _prepare_extra_models(self):
        self.codebook = torch.nn.Embedding(
            self._hashing._hash_size,
            self._data.training.shape[1],
        ).cuda()

    def _get_extra_models_parameters(self):
        return [p for p in self.codebook.parameters()]
