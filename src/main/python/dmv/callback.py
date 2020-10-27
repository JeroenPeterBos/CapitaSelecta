from datetime import datetime
from typing import Dict

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

import logging
logger = logging.getLogger(__name__)


class EpochLogger(Callback):
    def __init__(self):
        super().__init__()
        self._start_time = datetime.now()

    def on_epoch_begin(self, epoch, logs=None):
        self._start_time = datetime.now()

    def _time_since_epoch_start(self):
        return datetime.now() - self._start_time

    def on_epoch_end(self, epoch, logs: Dict[str, float] = None):
        title = f'Epoch {epoch} report: ({self._time_since_epoch_start()})'

        t, v = dict(), dict()
        for name, value in logs.items():
            if name.startswith('val_'):
                v[name[4:]] = value
            else:
                t[name] = value

        train, val = 'Training:     ', 'Validation:   '
        for name in list(set(t.keys()) & set(v.keys())) + list(set(t.keys()) ^ set(v.keys())):
            if name in t:
                train += '{}:  {:8} | '.format(name, '{:.4}'.format(t[name]))
            if name in v:
                val += '{}:  {:8} | '.format(name, '{:.4}'.format(v[name]))

        report = '\n'.join([title, train, val])

        logger.info(report)
