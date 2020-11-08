from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Masking
from tensorflow.python.layers.base import Layer
import tensorflow.python.keras.backend as K
import tensorflow as tf


import logging
logger = logging.getLogger(__name__)


class DynamicMultiViewMeanCell(Layer):
    def build(self, input_shape):
        self.units = input_shape[-1]
        self.built = True

    @property
    def state_size(self):
        return (self.units, 1)

    def call(self, inputs, states):
        state, count = states

        new_state = tf.math.add(inputs, state)
        new_count = tf.math.add(count, tf.constant([1], dtype=K.floatx()))

        output = tf.math.divide(new_state, new_count)
        return output, (new_state, new_count)


class DynamicMultiViewSumCell(Layer):
    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple):
            self.masked = False
            self.units = input_shape[-1]
        else:
            self.masked = True
            self.units = input_shape[0][-1]
        self.built = True

    @property
    def state_size(self):
        return self.units

    def call(self, inputs, states):
        state = states[0]

        if self.masked:
            inputs, mask = inputs
            new_state = tf.math.add(tf.math.multiply(inputs, mask), state)
        else:
            new_state = tf.math.add(inputs, state)

        return new_state, new_state


class DynamicMultiViewMaxCell(Layer):
    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple):
            self.masked = False
            self.units = input_shape[-1]
        else:
            self.masked = True
            self.units = input_shape[0][-1]
        self.built = True

    @property
    def state_size(self):
        return self.units

    def call(self, inputs, states):
        state = states[0]

        if self.masked:
            inputs, mask = inputs
            inputs = tf.math.multiply(inputs, mask)

        output = K.maximum(inputs, state)
        return output, output


class MultiViewDecisionLevelFusionModel(Model):
    def __init__(self, num_classes, input_shape, aggregation_type, masked):
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.aggregation_type = aggregation_type

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.classify = Dense(num_classes, activation='sigmoid', name='classify')
        self.agg = RNN(cell=self._translate_cell(aggregation_type)())

    @staticmethod
    def _translate_cell(cell):
        return {
            'mean': DynamicMultiViewMeanCell,
            'sum': DynamicMultiViewSumCell,
            'max': DynamicMultiViewMaxCell
        }[cell]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'aggregation_type': self.aggregation_type
        })
        logger.debug(f'dlf: {config}')
        return config

    def call(self, inputs, **kwargs):
        x = inputs
        if self.masked:
            x = Masking()(x)

        x = TimeDistributed(self.base)(inputs)
        x = TimeDistributed(self.classify)(x)

        x = self.agg(x)
        return x


class Max(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='max', masked=False)


class MaxMasked(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='max', masked=True)


class Mean(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=False)


class MeanMasked(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=True)
