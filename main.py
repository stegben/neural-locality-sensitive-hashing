import os
import sys
import argparse
from datetime import datetime
import random

from dotenv import load_dotenv

from encoders import TwoLayer256Relu
from nlsh.hashings import MultivariateBernoulli, Categorical
from nlsh.data import Glove
from nlsh.loggers import TensorboardX
from nlsh.trainers import TripletTrainer

from nlsh.learning.datasets import KNearestNeighborTriplet
from nlsh.learning.distances import L2, JSD_categorical

load_dotenv()

LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
MODEL_SAVE_DIR = os.environ["NLSH_MODEL_SAVE_DIR"]


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
        "--log_tag",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-tm",
        "--triplet_margin",
        type=float,
        default=10.0,
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
    return parser


def main():
    parser = nlsh_argparse()
    args = parser.parse_args()

    # hyper params
    k = args.k
    hash_size = args.hash_size
    lambda1 = args.lambda1
    triplet_margin = args.triplet_margin
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    data = Glove(os.environ.get("NLSH_PROCESSED_GLOVE_25_PATH"))
    data.load()
    enc = TwoLayer256Relu(input_dim=data.dim).cuda()
    hashing = MultivariateBernoulli(enc, hash_size, L2)
    # hashing = Categorical(enc, 2**HASH_SIZE, JSD_categorical)

    RUN_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_tag is None:
        RUN_NAME = f"{int(2**hash_size)}_triplet_{RUN_TIME}"
    logger = TensorboardX(f"{LOG_BASE_DIR}/{RUN_NAME}", RUN_NAME)
    logger.meta(params={
        'hash_size': hash_size,
        'lambda1': lambda1,
        'triplet_margin': triplet_margin,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': "glove 25",
        'k': k,
        'code_distance': "2-norm",
    })
    logger.args(' '.join(sys.argv[1:]))
    nlsh = TripletTrainer(
        hashing,
        data,
        MODEL_SAVE_DIR,
        logger=logger,
        lambda1=lambda1,
        triplet_margin=triplet_margin,
    )

    nlsh.fit(
        K=k,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
