from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import dmv.models as dmv_models


parser = ArgumentParser(description='Run a dynamic multi view research experiment.')

# The model to run
parser.add_argument(
    'model',
    type=str,
    help='The python directory path to the model to run the experiments on.'
)

# Run id to create the path
parser.add_argument(
    'run_id',
    type=str,
    help='The unique describing id for the type of run.'
)

# The location of the MURA data
parser.add_argument(
    '--data',
    type=Path,
    default=Path('/') / 'home' / 's1683047' / 'capita' / 'data' / 'full',
    help='Path to the MURA dataset.')

# The location of the log data
parser.add_argument(
    '--logs',
    type=Path,
    default=Path('/') / 'home' / 's1683047' / 'capita' / 'logs' / 'phase_1',
    help='The base path to the logs.'
)

# The category to filter the dataset on
parser.add_argument(
    '--category',
    type=str,
    default=None,
    help='The category to use for training, excluding all the others.'
)

# The image resolution to train the network on
parser.add_argument(
    '--img-size',
    type=int,
    nargs=3,
    default=[320, 320, 3],
    help='The image resolution to train the network on.'
)

# The single view and the multi view batch_size
parser.add_argument(
    '--batch-size',
    type=int,
    nargs=2,
    default=[8, 4],
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
    default=8,
    help='The maximum number of images a study is allowed to contain.'
)

# Maximum number of epochs
parser.add_argument(
    '--max-epochs',
    type=int,
    default=100,
    help='The maximum number of epochs to train the model for if early stopping doesn\'t kick in.'
)

# Replication
parser.add_argument(
    '--replication',
    type=int,
    default=5,
    help='The number of times to replicate the experiment.'
)

# shuffle size in samples
parser.add_argument(
    '--shuffle-size',
    type=int,
    default=-1,
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
    '--checkpoint',
    action='store_true',
    help='Checkpoint the best model at the end of the epoch.'
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
    help_message = parser.format_help()
    args = parser.parse_args()
    args.multi = 'multi' in args.model
    args.model = get_model_from_str(args.model)

    args.logs = args.logs / ('multi' if args.multi else 'single') / args.run_id / f'{datetime.now().strftime("%m-%d_%H-%M")}'

    return args, help_message


if __name__ == '__main__':
    import pprint
    args, help_message = parse_args()
    print(help_message)
    pprint.pprint(vars(args))
