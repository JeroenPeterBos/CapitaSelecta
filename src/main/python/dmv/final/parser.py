from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import dmv.models as dmv_models


parser = ArgumentParser(description='Run a dynamic multi view research experiment.')

parser.add_argument(
    'model_types',
    type=str,
    help="Which model types to run on, either 'multi' or 'single'."
)

# The location of the MURA data
parser.add_argument(
    '--data',
    type=Path,
    default=Path('/') / 'home' / 's1683047' / 'capita' / 'data' / 'folded',
    help='Path to the MURA dataset.')

# The location of the log data
parser.add_argument(
    '--logs',
    type=Path,
    default=Path('/') / 'home' / 's1683047' / 'capita' / 'logs' / 'final-128',
    help='The base path to the logs.'
)

parser.add_argument(
    '--folds',
    type=int,
    nargs='+',
    default=[0, 1, 2],
    help='The folds to try to load.'
)

# The image resolution to train the network on
parser.add_argument(
    '--img-size',
    type=int,
    nargs=3,
    default=[128, 128, 3],
    help='The image resolution to train the network on.'
)

# The single view and the multi view batch_size
parser.add_argument(
    '--batch-size',
    type=int,
    nargs=2,
    default=[32, 16],
    help='The single and multi view batch_size.'
)

# Caching imgs
parser.add_argument(
    '--cache-data',
    action='store_true',
    help='Whether to cache the images just before augmentation.'
)

# Maximum number of images a study is allowed to contain
parser.add_argument(
    '--max-imgs',
    type=int,
    default=12,
    help='The maximum number of images a study is allowed to contain.'
)

# Maximum number of epochs
parser.add_argument(
    '--max-epochs',
    type=int,
    default=100,
    help='The maximum number of epochs to train the model for if early stopping doesn\'t kick in.'
)

# shuffle size in samples
parser.add_argument(
    '--shuffle-size',
    type=int,
    default=1000,
    help='Shuffle size in samples. (-1 for full shuffle)'
)

# Redirect all the logs, also the tensorflow cpp logs to a file
parser.add_argument(
    '--redirect-err',
    action='store_true',
    help='Redirect all the logs (also .cc/cpp) to a file, the stderr will be hijacked so the stream will be gone.'
)

parser.add_argument(
    '--tensorboard',
    action='store_true',
    help='Store results in tensorboard files'
)

parser.add_argument(
    '--learning-rate',
    type=float,
    default=0.00001,
    help='The learning rate of the optimizer'
)


def parse_args():
    help_message = parser.format_help()
    args = parser.parse_args()
    return args, help_message


if __name__ == '__main__':
    import pprint
    args, help_message = parse_args()
    print(help_message)
    pprint.pprint(vars(args))
