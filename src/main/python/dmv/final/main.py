import logging
import multiprocessing
import os
import platform
import psutil
import re
import socket
import uuid
import sys
import io
import pprint
import random

from dmv.final.experiment import load_data, train_model, evaluate_model, predict_model
from dmv.final.parser import parse_args
import dmv.final.settings as settings

from tensorflow.python.client import device_lib
import tensorflow as tf

import numpy as np

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
        info['mac-address'] = ':'.join(re.findall('../..', '%012x' % uuid.getnode()))
        info['processor'] = platform.processor()
        info['ram'] = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
        info['cores'] = multiprocessing.cpu_count()
        info['gpus'] = get_available_gpus()
        info['python'] = sys.executable
        return info
    except Exception as e:
        logging.exception(e)


def setup_logger(log_dir, redirect_err, model_types):
    log_file = log_dir / f'out_{model_types}.log'
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

    if redirect_err:
        log_err = log_dir / f'err_{model_types}.log'
        sys.stderr.flush()
        err = open(log_err, 'a+')
        os.dup2(err.fileno(), sys.stderr.fileno())


def describe_environment(args):
    system = io.StringIO()
    params = io.StringIO()

    pprint.pprint(getSystemInfo(), stream=system)
    pprint.pprint(vars(args), stream=params)

    logger.info(f"The system we are running on: \n{system.getvalue()}")
    logger.info(f"The parameters describing the run are :\n{params.getvalue()}")


def fix_randomization():
    np.random.seed(15)
    tf.random.set_seed(15)
    random.seed(15)
    os.environ['PYTHONHASHSEED'] = '15'


if __name__ == '__main__':
    args, help_message = parse_args()
    setup_logger(args.logs, args.redirect_err, args.model_types)
    describe_environment(args)
    fix_randomization()

    logger.info(f'Usage: \n\n{help_message}')

    if args.model_types == 'single':
        models = settings.SINGLE_MODELS
    elif args.model_types == 'multi':
        models = settings.MULTI_MODELS
    else:
        raise Exception()

    logger.info(f"The models that will be trained are: {[str(m) for m in models]}")
    logger.info(f"The categories that will be evaluated are: {settings.CATEGORIES}")
    logger.info(f"These will be evaluated using several fold cross-validation, this run wil calculate folds {args.folds}.")

    for category in settings.CATEGORIES:
        logger.info(f"Running experiments for category: {category}")
        test_dc = load_data(args, 'test.csv', category, multi=True)

        for f in args.folds:
            train_dc = load_data(args, f'fold_{f}_train.csv', category, multi=args.model_types == 'multi')
            valid_dc = load_data(args, f'fold_{f}_val.csv', category, multi=True)

            for model_class in models:
                logger.info(f"Training model: {model_class.folder_id()}")
                log_dir = args.logs / category / model_class.folder_id() / f'fold_{f}'
                logger.info(f"Evaluating on fold {f}, the results are logged to {log_dir}")

                log_dir.mkdir(parents=True, exist_ok=True)

                model = train_model(args, model_class, train_dc, valid_dc, log_dir)
                evalution = evaluate_model(model, test_dc, log_dir)
                prediction = predict_model(model, test_dc, log_dir)

    logger.info("Finished")
