import os
import sys
import argparse
from datetime import datetime
import random

from dotenv import load_dotenv

from encoders import MultiLayerRelu, Siren
import nlsh
from nlsh.hashings import MultivariateBernoulli, Categorical
from nlsh.data import Glove, SIFT
from nlsh.loggers import TensorboardX, CometML, WandB, NullLogger
from nlsh.trainers import (
    TripletTrainer,
    SiameseTrainer,
    VQVAE,
    ProposedTrainer,
    AE,
    HierarchicalNavigableSmallWorldGraph,
)

from nlsh.learning.distances import (
    JSD_categorical,
    MVBernoulliL2,
    MVBernoulliKLDivergence,
    MVBernoulliCrossEntropy,
    MVBernoulliTanhCosine,
)

load_dotenv()

LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
MODEL_SAVE_DIR = os.environ["NLSH_MODEL_SAVE_DIR"]

COMET_API_KEY = os.environ["NLSH_COMET_API_KEY"]
COMET_PROJECT_NAME = os.environ["NLSH_COMET_PROJECT_NAME"]
COMET_WORKSPACE = os.environ["NLSH_COMET_WORKSPACE"]


def get_data_by_id(data_id):
    data_setting = data_id.split("_")
    if data_setting[0] == "glove":
        glove_dim = data_setting[1]
        assert glove_dim in ["25", "50", "100", "200"]
        path = os.environ.get(f"NLSH_PROCESSED_GLOVE_{glove_dim}_PATH")

        unit_norm = "norm" in data_id
        unit_ball = "sphere" in data_id
        return Glove(path, unit_norm, unit_ball)
    elif data_setting[0] == "sift":
        path = os.environ.get(f"NLSH_PROCESSED_SIFT_PATH")
        return SIFT(path, unit_norm="norm" in data_id)
    raise RuntimeError


def comma_separate_ints(value):
    try:
        str_ints = value.split(",")
        ints = [int(i) for i in str_ints]
        return ints
    except:
        msg = f"{value} is not a valid encoder structure." \
               "Should be comma separated integers, e.g. '256,256'"
        raise argparse.ArgumentTypeError(msg)


def hashing_type(value):
    allowed_hashings = ["Categorical", "MultivariateBernoulli"]
    if value not in allowed_hashings:
        msg = f"{value} is not a valid hashing type." \
              f"Only {', '.join(allowed_hashings)} are allowed"
        raise argparse.ArgumentTypeError(msg)
    return value


def get_hashing_from_args(args, enc):
    hashing_type = args.hashing_type
    distance_type = args.distance_type

    if hashing_type == "Categorical":
        # hash_size = int(2 ** args.hash_size)
        # if distance_type == "L2":
        #     return Categorical(enc, hash_size, L2)
        # elif distance_type == "JS":
        #     return Categorical(enc, hash_size, JSD_categorical)
        # else:
        #     raise RuntimeError(f"{distance_type} is not valid for {hashing_type}")
        raise RuntimeError("Categorical hashing not available temporarily")

    elif hashing_type == "MultivariateBernoulli":
        hash_size = args.hash_size
        if distance_type == "L2":
            return MultivariateBernoulli(
                enc,
                hash_size,
                MVBernoulliL2(),
            )
        elif distance_type == "KL":
            return MultivariateBernoulli(
                enc,
                hash_size,
                MVBernoulliKLDivergence(epsilon=1e-20),
            )
        elif distance_type == "CrossEntropy":
            return MultivariateBernoulli(
                enc,
                hash_size,
                MVBernoulliCrossEntropy(epsilon=1e-20),
            )
        else:
            raise RuntimeError(f"{distance_type} is not valid for {hashing_type}")

    elif hashing_type == "MultivariateBernoulliTanh":
        hash_size = args.hash_size
        if distance_type == "Cosine":
            return MultivariateBernoulli(
                enc,
                hash_size,
                MVBernoulliTanhCosine(),
                tanh_output=True,
            )
        else:
            raise RuntimeError(f"{distance_type} is not valid for {hashing_type}")

    else:
        raise RuntimeError(f"{hashing_type} is not a valid hashing type")


def get_logger_from_args(args):
    if args.debug:
        logger = NullLogger()
    else:
        if args.logger_type == "cometml":
            log_tags = args.log_tags
            log_tags = log_tags.split(",") if log_tags is not None else None
            logger = CometML(
                api_key=COMET_API_KEY,
                project_name=COMET_PROJECT_NAME,
                workspace=COMET_WORKSPACE,
                debug=args.debug,
                tags=log_tags,
            )
        elif args.logger_type == "tensorboard":
            run_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{int(2**args.hash_size)}_triplet_{run_time}"
            logger = TensorboardX(f"{LOG_BASE_DIR}/{run_name}", run_name)
        elif args.logger_type == "wandb":
            log_tags = args.log_tags
            log_tags = log_tags.split(",") if log_tags is not None else None
            logger = WandB(log_tags)
        else:
            raise RuntimeError(f"{args.logger_type} is not a valid logger type")

    logger.meta(params={
        'k': args.k,

        # hash function related
        'hash_size': args.hash_size,
        'encoder_structure': args.encoder_structure,
        'distance_type': args.distance_type,

        # data related
        'data_id': args.data_id,

        # fitting related
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
    })
    logger.args(' '.join(sys.argv[1:]))
    return logger


def get_learner_from_args(args, hashing, data, logger):
    if args.learner_type == "triplet":
        lambda1 = args.lambda1
        margin = args.triplet_margin
        triplet_positive_k = args.triplet_positive_k
        triplet_negative_sampling_method = args.triplet_negative_sampling_method
        logger.meta(params={
            "learner_type": "triplet",
            "learner_args": f"m={margin} l1={lambda1} pk={triplet_positive_k}",
            "triplet_margin": margin,
            "triplet_positive_k": triplet_positive_k,
            "triplet_negative_sampling_method": triplet_negative_sampling_method,
            "lambda1": lambda1,
        })
        learner = TripletTrainer(
            hashing,
            data,
            MODEL_SAVE_DIR,
            logger=logger,
            lambda1=lambda1,
            margin=margin,
            positive_k=triplet_positive_k,
            negative_sampling_method=triplet_negative_sampling_method,
        )
    elif args.learner_type == "siamese":
        lambda1 = args.lambda1
        positive_margin = args.siamese_positive_margin
        negative_margin = args.siamese_negative_margin
        positive_rate = args.siamese_positive_rate
        logger.meta(params={
            "learner_type": "siamese",
            "learner_args": f"nm={negative_margin} pm={positive_margin} pr={positive_rate}",
            'siamese_positive_margin': positive_margin,
            'siamese_negative_margin': negative_margin,
            'siamese_positive_rate': positive_rate,
            "lambda1": lambda1,
        })
        learner = SiameseTrainer(
            hashing,
            data,
            MODEL_SAVE_DIR,
            logger=logger,
            lambda1=lambda1,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
            positive_rate=positive_rate,
        )
    elif args.learner_type == "vqvae":
        logger.meta(params={
            "learner_type": "vqvae",
        })
        learner = VQVAE(
            hashing,
            data,
            MODEL_SAVE_DIR,
            logger=logger,
        )
    elif args.learner_type == "proposed":
        lambda1 = args.lambda1
        logger.meta(params={
            "learner_type": "proposed",
            "learner_args": f"train_k=10 l1={lambda1}",
        })
        learner = ProposedTrainer(
            hashing,
            data,
            MODEL_SAVE_DIR,
            logger=logger,
            train_k=10,
            lambda1=lambda1,
        )
    elif args.learner_type == "ae":
        logger.meta(params={
            "learner_type": "ae",
        })
        learner = AE(
            hashing,
            data,
            MODEL_SAVE_DIR,
            logger=logger,
        )
    elif args.learner_type == "hnsw":
        logger.meta(params={
            "learner_type": "hnsw",
        })
        learner = HierarchicalNavigableSmallWorldGraph(
            data,
            logger=logger,
        )
    return learner


def nlsh_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-hs",
        "--hash_size",
        type=int,
        default=12,
    )
    parser.add_argument(
        "-es",
        "--encoder_structure",
        type=comma_separate_ints,
        default='256,256',
    )
    parser.add_argument(
        "-ht",
        "--hashing_type",
        default='MultivariateBernoulli',
        choices=("MultivariateBernoulli", "MultivariateBernoulliTanh", "Categorical"),
    )
    parser.add_argument(
        "-dt",
        "--distance_type",
        default='L2',
        choices=("L2", "JS", "KL", "CrossEntropy", "Cosine"),
    )
    parser.add_argument(
        "--data_id",
        # choices=("glove_25", "glove_50", "glove_100", "glove_200",),
    )
    parser.add_argument(
        "--logger_type",
        choices=("tensorboard", "cometml", "wandb"),
    )
    parser.add_argument(
        "--log_tags",
        default=None,
    )
    parser.add_argument(
        "--learner_type",
        choices=("triplet", "siamese", "vqvae", "proposed", "ae", "hnsw"),
    )
    parser.add_argument(
        "-tm",
        "--triplet_margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-tpk",
        "--triplet_positive_k",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-tnsm",
        "--triplet_negative_sampling_method",
        type=str,
        default="random",
        choices=("random", "nearest"),
    )
    parser.add_argument(
        "-spm",
        "--siamese_positive_margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-snm",
        "--siamese_negative_margin",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-spr",
        "--siamese_positive_rate",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=2e-2,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    return parser


def main():
    parser = nlsh_argparse()
    args = parser.parse_args()

    # hyper params
    k = args.k
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    print("=== read data ===")
    data = get_data_by_id(args.data_id)
    data.load()
    print("=== prepare encoder ===")
    # enc = MultiLayerRelu(
    enc = Siren(
        input_dim=data.dim,
        hidden_dims=args.encoder_structure,
    ).cuda()
    hashing = get_hashing_from_args(args, enc)
    logger = get_logger_from_args(args)
    print("=== prepare learner ===")
    learner = get_learner_from_args(args, hashing, data, logger)

    print("Start training")
    learner.fit(
        K=k,
        batch_size=batch_size,
        learning_rate=learning_rate,
        test_every_updates=300,
    )


if __name__ == '__main__':
    main()
