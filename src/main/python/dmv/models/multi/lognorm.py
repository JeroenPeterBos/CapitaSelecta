from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN

from dmv.layer import Mask, DynamicMultiViewRNN, LogNormCell

import logging
logger = logging.getLogger(__name__)


class LogNormModel(Model):
    def __init__(self, num_classes, input_shape, masked):
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.classify = Dense(num_classes, activation=None, name='classify')
        self.agg = RNN(LogNormCell())

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
        })
        return config

    def call(self, inputs, **kwargs):
        x = TimeDistributed(self.base)(inputs)
        x = TimeDistributed(self.classify)(x)

        if self.masked:
            y = TimeDistributed(Mask())(inputs)
            x = self.agg((x, y))
        else:
            x = self.agg(x)
        return x


class Base(LogNormModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=False)


class Masked(LogNormModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True)
