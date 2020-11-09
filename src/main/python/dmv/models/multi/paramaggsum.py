from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Layer, Masking, MaxPooling2D
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
            shape=(self._input_neurons,),
            initializer='ones',
            name='weights'
        )
        self._q = self.add_weight(
            shape=(self._input_neurons,),
            initializer='zeros',
            name='weights'
        )
        self._bias = self.add_weight(
            shape=(self._input_neurons,),
            initializer='zeros',
            name='bias'
        )

    @property
    def state_size(self):
        return [tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons])]

    @property
    def output_size(self):
        return [tf.TensorShape([self._input_neurons])]

    def call(self, inputs, states):
        state, count = states
        x = inputs

        if self._pre_activation is not None:
            y = tf.math.multiply(x, tf.constant([self._beta], dtype=K.floatx()))
            x = tf.math.multiply(self._pre_activation(x), tf.constant([1 - self._beta], dtype=K.floatx()))
            x = tf.math.add(x, y)

        # Aggregate
        c = tf.math.add(count, tf.constant([1], dtype=K.floatx()), name='count_increment')
        s = tf.math.add(x, state, name='sum_state')

        # Log transform and apply dense
        x = s
        x = tf.math.multiply(x, self._w, 'multi_weights')
        x = tf.math.add(x, tf.math.multiply(self._q, c))
        x = tf.math.add(x, self._bias, name='add_bias')

        x = self._post_activation(x)

        return x, (s, c)

    def get_config(self):
        return {
            "pre_activation": self._pre_activation,
            "post_activation": self._post_activation,
            "beta": self._beta
        }


class ParamAggModel(Model):
    def __init__(self, num_classes, input_shape, masked, pre_activation=None):
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.pre_activation = pre_activation

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.agg = RNN(ParamAggCell(pre_activation=pre_activation))
        self.classify = Dense(num_classes, activation='sigmoid', name='classify')

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

        x = TimeDistributed(self.base)(x)
#        x = TimeDistributed(MaxPooling2D())(x)

        x = self.agg(x)
        x = self.classify(x)
        return x


class Sum(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True)


class Sigmoid(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True, pre_activation='sigmoid')


class Relu(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True, pre_activation='relu')


class Tanh(ParamAggModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, masked=True, pre_activation='tanh')