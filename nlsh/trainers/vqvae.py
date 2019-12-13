from time import time

import torch
import torch.nn.functional as F
import numpy as np

from nlsh.metrics import calculate_recall
from nlsh.indexer import Indexer


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


class VQVAE:

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

    def fit(self, K, batch_size=1024, learning_rate=3e-4, test_every_updates=1000):
        if not self._data.prepared:
            self._data.load()
        candidate_vectors = self._data.training
        validation_data = self._data.testing
        ground_truth = self._data.ground_truth[:, :K]

        self._candidate_vectors = torch.from_numpy(candidate_vectors)
        self._candidate_vectors_gpu = torch.from_numpy(candidate_vectors).cuda()
        self._validation_data = torch.from_numpy(validation_data)
        self._validation_data_gpu = self._validation_data.cuda()

        dataset = OnePass(self._candidate_vectors_gpu)
        codebook = torch.nn.Embedding(self._hashing._hash_size, self._data.training.shape[1]).cuda()
        optimizer = torch.optim.SGD(
            [p for p in self._hashing.parameters()] + [p for p in codebook.parameters()],
            # codebook.parameters(),
            lr=learning_rate,
            # amsgrad=True,
        )

        global_step = 0
        best_recall = 0.
        for _ in range(300):
            for sampled_batch in dataset.batch_generator(batch_size, True):
                global_step += 1

                self._hashing.train_mode(True)
                optimizer.zero_grad()
                probs = self._hashing.predict(sampled_batch)
                probs.retain_grad()
                codes = StraightThroughCodebookLookup.apply(probs, codebook.weight)
                codes.retain_grad()
                dist = F.pairwise_distance(codes, sampled_batch)
                loss = (dist**2).mean()

                # TODO: potential uniform regularizers
                # loss += self._lambda1 * torch.mm(anchor, anchor.T).max(1)[0].mean()
                # loss -= self._lambda1 * torch.log(torch.cdist(anchor, anchor).topk(2, dim=1, largest=False)[0][:,1]).mean()

                self._logger.log("training/loss", loss.data.cpu(), global_step)
                loss.backward()
                # import ipdb; ipdb.set_trace()
                optimizer.step()
                # if global_step % 1 == 0:
                if global_step % test_every_updates == 0:
                    # import ipdb; ipdb.set_trace()
                    self._hashing.train_mode(False)
                    # import ipdb; ipdb.set_trace()
                    indexer = Indexer(
                        self._hashing,
                        self._candidate_vectors,
                        self._candidate_vectors_gpu,
                        self._data.distance,
                    )
                    n_indexes = len(indexer.index2row)
                    self._logger.log("test/n_indexes", n_indexes, global_step)
                    std_index_rows = np.std([len(idxs) for idxs in indexer.index2row.values()])
                    self._logger.log("test/std_index_rows", std_index_rows, global_step)

                    t1 = time()
                    result = indexer.query(self._validation_data, self._validation_data_gpu, k=K)
                    t2 = time()
                    query_time = t2 - t1
                    current_recall = calculate_recall(list(ground_truth), result, np.mean)

                    if current_recall > best_recall:
                        base_name = f"{self._model_save_dir}/{self._logger.run_name}_{global_step}_{current_recall:.4f}"
                        self._hashing.save(base_name)
                        best_recall = current_recall

                    self._logger.log("test/recall", current_recall, global_step)
                    qps = self._validation_data.shape[0] / query_time
                    self._logger.log("test/qps", qps, global_step)


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
