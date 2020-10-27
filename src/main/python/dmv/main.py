from datetime import datetime
from pathlib import Path

from dmv.parser import parse_args
from dmv.experiment import load_data, experiment

import platform, socket, re, uuid, json, psutil, logging, multiprocessing

logger = logging.getLogger()


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

    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)


def describe_environment(args):
    logger.info(f"The system we are running on: \n\n{getSystemInfo()}")
    logger.info(f"The parameters describing the run are :\n\n{vars(args)}")


def main(args):
    setup_logger(args.logs)
    describe_environment(args)
    logger.info("Starting a new experiment")

    train_dc, valid_dc = load_data(
        data_folder=args.data,
        multi=args.multi,
        img_shape=args.img_size,
        batch_size=args.batch_size,
        category=args.category,
        cache_imgs=args.cache_imgs,
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


if __name__ == '__main__':
    args = parse_args()
    main(args)
