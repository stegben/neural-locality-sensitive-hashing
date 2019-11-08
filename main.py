import os
import sys
import argparse
from datetime import datetime
import random

from dotenv import load_dotenv

from encoders import MultiLayerRelu
from nlsh.hashings import MultivariateBernoulli, Categorical
from nlsh.data import Glove
from nlsh.loggers import TensorboardX, CometML
from nlsh.trainers import TripletTrainer

from nlsh.learning.datasets import KNearestNeighborTriplet
from nlsh.learning.distances import (
    L2,
    JSD_categorical,
)

load_dotenv()

LOG_BASE_DIR = os.environ["NLSH_TENSORBOARD_LOG_DIR"]
MODEL_SAVE_DIR = os.environ["NLSH_MODEL_SAVE_DIR"]

COMET_API_KEY = os.environ["NLSH_COMET_API_KEY"]
COMET_PROJECT_NAME = os.environ["NLSH_COMET_PROJECT_NAME"]
COMET_WORKSPACE = os.environ["NLSH_COMET_WORKSPACE"]


def get_data_by_id(data_id):
    id2path = {
        "glove_25": os.environ.get("NLSH_PROCESSED_GLOVE_25_PATH"),
        "glove_50": os.environ.get("NLSH_PROCESSED_GLOVE_50_PATH"),
        "glove_100": os.environ.get("NLSH_PROCESSED_GLOVE_100_PATH"),
        "glove_200": os.environ.get("NLSH_PROCESSED_GLOVE_200_PATH"),
    }
    return Glove(id2path[data_id])


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
    if hashing_type == "Categorical":
        hash_size = int(2 ** args.hash_size)
        return Categorical(enc, hash_size, JSD_categorical)
    elif hashing_type == "MultivariateBernoulli":
        hash_size = args.hash_size
        return MultivariateBernoulli(enc, hash_size, L2)
    else:
        raise RuntimeError(f"{hashing_type} is not a valid hashing type")


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
        type=str,
        default='multivariate_bernoulli',
    )
    parser.add_argument(
        "--data_id",
        type=str,
        choices=("glove_25", "glove_50", "glove_100", "glove_200",),
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
    encoder_structure = args.encoder_structure
    lambda1 = args.lambda1
    triplet_margin = args.triplet_margin
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    data = get_data_by_id(args.data_id)
    data.load()
    enc = MultiLayerRelu(
        input_dim=data.dim,
        hidden_dims=encoder_structure,
    ).cuda()
    hashing = get_hashing_from_args(args, enc)

    RUN_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_tag is None:
        RUN_NAME = f"{int(2**hash_size)}_triplet_{RUN_TIME}"
    # logger = TensorboardX(f"{LOG_BASE_DIR}/{RUN_NAME}", RUN_NAME)
    logger = CometML(
        api_key=COMET_API_KEY,
        project_name=COMET_PROJECT_NAME,
        workspace=COMET_WORKSPACE,
        debug=False,
    )
    logger.meta(params={
        'hash_size': hash_size,
        'encoder_structure': encoder_structure,
        'lambda1': lambda1,
        'triplet_margin': triplet_margin,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'data': "glove 25",
        'k': k,
        'code_distance': "2-norm",
        'data_id': args.data_id,
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
