import abc


class _Trainer(abc.ABC):

    def __init__(
            self,
            candidate_vectors,
            learnable_hash_functions,
            distance_function,
            code_distance_function,
            logger,
        ):
        pass

    def fit(self):
        pass


class ContrastiveLoss(_Trainer):
    pass


class TripletLoss(_Trainer):
    pass
