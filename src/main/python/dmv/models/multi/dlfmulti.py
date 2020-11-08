from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Masking

from dmv.layer import Mask, DmvCell

import tensorflow as tf
import tensorflow.keras.backend as K

import logging
logger = logging.getLogger(__name__)


class DmvDlfModel(Model):
    def __init__(self,
                 num_classes,
                 input_shape,
                 aggregations,
                 masked=True):
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.aggregations = aggregations

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True
        self.base = TimeDistributed(self.base)

        self.condense = TimeDistributed(Dense(num_classes, activation='sigmoid', name='condense'))
        self.agg = RNN(DmvCell(aggregations=aggregations))
        #self.classify_1 = Dense(4, activation='relu', name='classify_1')
        #self.classify = Dense(num_classes, activation='sigmoid', name='classify')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'aggregations': self.aggregations
        })
        return config

    def call(self, inputs, **kwargs):
        x = inputs

        if self.masked:
            x = Masking()(x)

        x = self.base(x)
        x = self.condense(x)

        x = self.agg(x)
        #x = self.classify_1(x)
        #x = self.classify(x)

        return x


class Mean(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean'])


class Max(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['max'])


class MeanMax(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean', 'max'])


class MeanMaxStd(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean', 'max', 'std'])


class Std(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['std'])


class MeanStd(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean', 'std'])


class MeanVar(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean', 'var'])


class SquareMean(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['square_mean'])


class SquareSum(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['square_sum'])


class CustomSum(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['custom_sum'])


class CustomMean(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['custom_mean'])


class MeanMaxCombo(DmvDlfModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregations=['mean_max'])
