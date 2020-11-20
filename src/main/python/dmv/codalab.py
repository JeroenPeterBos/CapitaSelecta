import tensorflow as tf
import sys

import logging

from dmv import settings

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)

e = {
    'python': sys.version_info,
    'tensorflow': tf.__version__
}
logger.info(f"Environment: {e}")

from argparse import ArgumentParser
from pathlib import Path
from tensorflow.python.keras import Model
from dmv.data import DataContainer
import numpy as np

tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.DEBUG)
tf_logger.addHandler(ch)

this_folder = Path(__file__).parent.absolute()

parser = ArgumentParser(description='Run a trained model on Codalab')

parser.add_argument(
    'input_csv',
    type=Path,
    help='The path to the input csv'
)

parser.add_argument(
    'output_csv',
    type=Path,
    help='The path to the output csv'
)

parser.add_argument(
    'model',
    type=str,
    help='The python directory path to the model to run the experiments on.'
)

parser.add_argument(
    'weights',
    type=Path,
    help='The path to the weights.'
)

parser.add_argument(
    '--root',
    type=Path,
    default=None,
    help='The root path to all the images'
)

parser.add_argument(
    '--full-out',
    action='store_true',
    help='Output a detailed csv file instead of what codalab requires'
)

parser.add_argument(
    '--img-size',
    type=int,
    nargs=3,
    default=[320, 320, 3],
    help='The image resolution to train the network on.'
)


def get_model_from_str(python_path: str):
    """
    From: https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class

    Args:
        python_path: The path to the model from the models directory

    Returns:

    """
    package_str = 'dmv.models.' + '.'.join(python_path.split('.')[:-1])
    class_str = python_path.split('.')[-1]

    return getattr(__import__(package_str, fromlist=[class_str]), class_str)


def parse_args():
    args = parser.parse_args()
    args.model = get_model_from_str(args.model)
    args.weights = this_folder / args.weights
    args.root = args.root if args.root is not None else this_folder.parent
    return args


def load_data(meta_csv, root_folder, img_shape):
    common = {
        'batch_size': 1,
        'output_shape': list(img_shape),
        'max_imgs': 8
    }

    dc = DataContainer(root_folder, meta_csv=meta_csv, multi=True, train=False, **common)
    return dc


def run():
    args = parse_args()
    logger.info(f"Running a codalab evaluation for model of type {args.model}")

    settings.DENSENET_INIT = None

    logger.info(f"Loading the data in directory {args.root} using meta file {args.input_csv}")
    dc = load_data(meta_csv=args.input_csv, root_folder=args.root, img_shape=args.img_size)

    logger.info(f"Loading the weights in folder {args.weights}")
    model: Model = args.model(num_classes=1, input_shape=args.img_size)
    model.load_weights(args.weights)

    logger.info("Calculating the predictions")
    prediction = model.predict(dc.ds(), steps=dc.batches_per_epoch)

    logger.info(f"Logging the predictions to a file {args.output_csv}")
    df = dc.df
    df['predictions'] = np.round(prediction)

    if args.full_out:
        df.to_csv(args.output_csv)
    else:
        df[['folder', 'predictions']].to_csv(args.output_csv, header=False, index=False)

    logger.info("Finished")


if __name__ == '__main__':
    run()
