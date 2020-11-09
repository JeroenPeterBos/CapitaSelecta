from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Layer, Masking
from tensorflow.python.keras import activations
import tensorflow as tf
import tensorflow.keras.backend as K

import logging
logger = logging.getLogger(__name__)


class ParamAggCell(Layer):
    def __init__(self,
                 pre_activation=None,
                 post_activation='sigmoid',
                 beta=0.1,
                 **kwargs):
        self._pre_activation = activations.deserialize(pre_activation)
        self._post_activation = activations.deserialize(post_activation)
        self._beta = beta
        super().__init__(**kwargs)

    def build(self, input_shapes):
        self._input_neurons = input_shapes[-1]

        self._w = self.add_weight(
            shape=(1, 1),
            initializer='ones',
            name='weights'
        )
        self._q = self.add_weight(
            shape=(1, 1),
            initializer='zeros',
            name='std_weights'
        )
        self._bias = self.add_weight(
            shape=(1, 1),
            initializer='zeros',
            name='bias'
        )

    @property
    def state_size(self):
        return [tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons]), tf.TensorShape([1])]

    @property
    def output_size(self):
        return [tf.TensorShape([self._input_neurons])]

    def call(self, inputs, states):
        sum_state, mean_state, sn_state, count_state = states
        x = inputs

        if self._pre_activation is not None:
            y = tf.math.multiply(x, tf.constant([self._beta], dtype=K.floatx()))
            x = tf.math.multiply(self._pre_activation(x), tf.constant([1 - self._beta], dtype=K.floatx()))
            x = tf.math.add(x, y)

        # Calculate the new states and the mean
        count_new = tf.math.add(count_state, tf.constant([1], dtype=K.floatx()), name='count_increment')
        sum_new = tf.math.add(x, sum_state, name='sum_state')

        mean_new = tf.math.divide(sum_new, count_new, name='calc_mean')

        sn_new = tf.math.add(
            sn_state,
            tf.math.multiply(
                tf.math.subtract(x, mean_state, name='dmv_sn_sub_1'),
                tf.math.subtract(x, mean_new, name='dmv_sn_sub_2'),
                name='dmv_sn_mult'
            ),
            name='dmv_sn_add'
        )

        var_new = tf.math.divide(sn_new, count_new)

        # Calculating the output
        out_mean = tf.math.multiply(mean_new, self._w)
        out_var = tf.math.multiply(var_new, self._q)

        out = tf.math.add(out_mean, out_var)
        out = tf.math.add(out, self._bias, name='add_bias')

        out = self._post_activation(out)

        return out, (sum_new, mean_new, sn_new, count_new)

    def get_config(self):
        return {
            "pre_activation": self._pre_activation,
            "post_activation": self._post_activation,
            "beta": self._beta
        }


class ParamAggModel(Model):
    def __init__(self, num_classes, input_shape, masked, pre_activation=None, beta=0.1):
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.pre_activation = pre_activation
        self.beta = beta

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.classify = Dense(num_classes, activation=None, name='classify')
        self.agg = RNN(ParamAggCell(pre_activation=pre_activation, beta=beta))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'pre_activation': self.pre_activation,
            'beta': self.beta
        })
        return config

    def call(self, inputs, **kwargs):
        x = inputs
        if self.masked:
            x = Masking()(x)

        x = TimeDistributed(self.base)(x)
        x = TimeDistributed(self.classify)(x)

        x = self.agg(x)
        return x


class Base(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True)


class Sigmoid(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True, pre_activation='sigmoid')


class SigmoidBeta(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True, pre_activation='sigmoid', beta=0.2)
