import os
import argparse
from datetime import datetime
import random

from dotenv import load_dotenv

from encoders import TwoLayer256Relu
from nlsh.hashings import MultivariateBernoulli
from nlsh.data import Glove
from nlsh.loggers import TensorboardX
from nlsh.trainers import TripletTrainer

from nlsh.learning.datasets import KNearestNeighborTriplet
from nlsh.learning.distances import L2

load_dotenv()

K = 10
HASH_SIZE = 12
LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
MODEL_SAVE_DIR = os.environ["NLSH_MODEL_SAVE_DIR"]
RUN_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_NAME = f"{K}_{HASH_SIZE}_triplet_{RUN_TIME}"


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
        "-d",
        "--data",
        type=str,
    )
    parser.add_argument(
        "--distance",
        type=str,
    )
    parser.add_argument(
        "-l",
        "--learner",
        type=str,
    )
    parser.add_argument(
        "--log_tag",
        type=str,
        default="",
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=1e-2,
    )
    return parser


def main():
    parser = nlsh_argparse()
    args = parser.parse_args()

    data = Glove(os.environ.get("NLSH_PROCESSED_GLOVE_25_PATH"))
    data.load()
    enc = TwoLayer256Relu(input_dim=data.dim).cuda()
    hashing = MultivariateBernoulli(enc, HASH_SIZE, L2)
    logger = TensorboardX(f"{LOG_BASE_DIR}/{RUN_NAME}", RUN_NAME)

    nlsh = TripletTrainer(
        hashing,
        data,
        MODEL_SAVE_DIR,
        logger,
        lambda1=0.02,
        triplet_margin=1.0,
    )

    nlsh.fit(K=K)

    # TODO: model serialization
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
