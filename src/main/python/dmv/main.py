import logging
import multiprocessing
import platform
import psutil
import re
import socket
import uuid
import sys
import io
import pprint
from datetime import datetime
from pathlib import Path

from dmv.experiment import load_data, experiment
from dmv.parser import parse_args

from tensorflow.python.client import device_lib
import tensorflow as tf


logger = logging.getLogger()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def getSystemInfo():
    try:
        info = dict()
        info['platform'] = platform.system()
        info['platform-release'] = platform.release()
        info['platform-version'] = platform.version()
        info['architecture'] = platform.machine()
        info['hostname'] = socket.gethostname()
        info['ip-address'] = socket.gethostbyname(socket.gethostname())
        info['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        info['processor'] = platform.processor()
        info['ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
        info['cores'] = multiprocessing.cpu_count()
        info['gpus'] = get_available_gpus()
        info['python'] = sys.executable
        return info
    except Exception as e:
        logging.exception(e)


def setup_logger(log_dir):
    log_file: Path = log_dir / f'log_{datetime.now().strftime("%m-%d_%H-%M")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_file)

    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # Add handlers to the default logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Add handlers to the file handler
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.setLevel(logging.DEBUG)
    tf_logger.addHandler(ch)
    tf_logger.addHandler(fh)

    tf_logger_2 = tf.get_logger()
    tf_logger_2.setLevel(logging.DEBUG)
    tf_logger_2.addHandler(ch)
    tf_logger_2.addHandler(fh)


def describe_environment(args):
    system = io.StringIO()
    params = io.StringIO()

    pprint.pprint(getSystemInfo(), stream=system)
    pprint.pprint(vars(args), stream=params)

    logger.info(f"The system we are running on: \n{system.getvalue()}")
    logger.info(f"The parameters describing the run are :\n{params.getvalue()}")


def main(args):
    logger.info("Starting a new experiment")

    train_dc, valid_dc = load_data(
        data_folder=args.data,
        multi=args.multi,
        img_shape=args.img_size,
        batch_size=args.batch_size,
        category=args.category,
        cache_imgs=args.cache_data,
        max_imgs=args.max_imgs,
        shuffle_size=args.shuffle_size
    )

    for i in range(args.replication):
        log = args.logs / datetime.now().strftime("%m-%d_%H-%M")
        logger.info(f"___ Running replication {i} out of {args.replication} which will be logged to: '{log}'.")

        experiment(
            train_dc=train_dc,
            valid_dc=valid_dc,
            log_folder=log,
            model_class=args.model,
            max_epochs=args.max_epochs
        )

    logging.info("Finished experiment")


if __name__ == '__main__':
    args, help_message = parse_args()
    setup_logger(args.logs)
    describe_environment(args)

    logger.info(f'Usage: \n\n{help_message}')

    main(args)
