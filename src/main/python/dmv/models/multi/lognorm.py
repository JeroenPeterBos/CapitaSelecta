from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Layer, Masking
from tensorflow.python.keras import activations
import tensorflow as tf
import tensorflow.keras.backend as K

import logging
logger = logging.getLogger(__name__)


class LogNormCell(Layer):
    def __init__(self,
                 pre_activation='softplus',
                 post_activation='sigmoid',
                 extra_log_transforms=False,
                 **kwargs):
        self._pre_activation = activations.deserialize(pre_activation)
        self._post_activation = activations.deserialize(post_activation)
        self._extra_log_transforms=extra_log_transforms
        super().__init__(**kwargs)

    def build(self, input_shapes):
        self._input_neurons = input_shapes[-1]

        self._w = self.add_weight(
            shape=(self._input_neurons, 1),
            initializer='glorot_uniform',
            name='w',
        )
        self._q = self.add_weight(
            shape=(1, 1),
            initializer='glorot_uniform',
            name='q'
        )
        self._bias = self.add_weight(
            shape=(1, 1),
            initializer='zeros',
            name='bias'
        )

    @property
    def state_size(self):
        return [tf.TensorShape([self._input_neurons]), tf.TensorShape([1])]

    @property
    def output_size(self):
        return [tf.TensorShape([self._input_neurons])]

    def call(self, inputs, states):
        state, count = states

        x = self._pre_activation(inputs)

        # Aggregate
        c = tf.math.add(count, tf.constant([1], dtype=K.floatx()), name='count_increment')
        s = tf.math.add(x, state, name='sum_state')

        if self._extra_log_transforms:
            # TODO: implement?
            pass
        else:
            s_log = tf.math.log(s, name='x_log_transform')
            c_log = tf.math.log(c, name='c_log_transform')

        # Log transform and apply dense
        x = tf.math.subtract(
            tf.math.multiply(s_log, self._w, name='x_apply_weights'),
            tf.math.multiply(c_log, self._q, name='c_apply_weights'),
            name='dense_linear'
        )
        x = tf.math.add(
            x,
            self._bias,
            name='dense_bias'
        )

        x = self._post_activation(x)

        return x, (s, c)

    def get_config(self):
        return {
            "pre_activation": self._pre_activation,
            "post_activation": self._post_activation,
            "extra_log_transforms": self._extra_log_transforms
        }


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
        x = inputs
        if self.masked:
            x = Masking()(x)

        x = TimeDistributed(self.base)(inputs)
        x = TimeDistributed(self.classify)(x)

        x = self.agg(x)
        return x


class Base(LogNormModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=False)


class Masked(LogNormModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True)
