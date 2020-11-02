from tensorflow.keras.layers import AbstractRNNCell, RNN, Layer
import tensorflow.keras.activations as activations
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from dmv import layer_serialization


import logging
logger = logging.getLogger(__name__)


class Mask(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.built = True
    
    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape([None, 1])
    
    def call(self, x):
        x = tf.math.count_nonzero(x, axis=[1, 2, 3])
        x = tf.math.greater(tf.cast(x, dtype=K.floatx()), tf.constant([0], dtype=K.floatx()))
        x = tf.expand_dims(x, axis=-1)
        x = tf.cast(x, K.floatx())
        return x


class DynamicMultiViewMeanCell(AbstractRNNCell):
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
        return (self.units, 1)
    
    def call(self, inputs, states):
        state, count = states

        if self.masked:
            inputs, mask = inputs
            new_state = tf.math.add(tf.math.multiply(inputs, mask), state)
            new_count = tf.math.add(count, mask)
        else:
            new_state = tf.math.add(inputs, state)
            new_count = tf.math.add(count, tf.constant([1], dtype=K.floatx()))

        output = tf.math.divide(new_state, new_count)
        return output, (new_state, new_count)


class DynamicMultiViewSumCell(AbstractRNNCell):
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


class DynamicMultiViewMaxCell(AbstractRNNCell):
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


class DynamicMultiViewRNN(RNN):
    def __init__(self, aggregation_type: str = 'mean', **kwargs):
        self.aggregation_type = aggregation_type
        if aggregation_type == 'max':
            cell = DynamicMultiViewMaxCell(**kwargs)
        elif aggregation_type == 'mean':
            cell = DynamicMultiViewMeanCell(**kwargs)
        elif aggregation_type == 'sum':
            cell = DynamicMultiViewSumCell(**kwargs)
        super().__init__(cell)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'aggregation_type': self.aggregation_type
        })
        return config
    
    def call(self, inputs, **kwargs):
        return super().call(inputs, **kwargs)

    @property
    def _trackable_saved_model_saver(self):
        return layer_serialization.CustomRNNSavedModelSaver(self)


class DynamicMultiViewRNNCell2D(AbstractRNNCell):
    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shape):
        self.units = input_shape[-1]
        self.built = True
        
    @property
    def state_size(self):
        return self.units
    
    def call(self, inputs, states):
        output = K.maximum(inputs, states[0])
        return output, output


class DynamicMultiViewRNN2D(ConvRNN2D):
    def __init__(self, **kwargs):
        cell = DynamicMultiViewRNNCell2D(**kwargs)
        super().__init__(cell)
    
    def call(self, inputs):
        return super().call(inputs)


class LogNormCell(Layer):
    def __init__(self,
                 pre_activation='softplus',
                 post_activation='sigmoid',
                 **kwargs):
        self._pre_activation = activations.deserialize(pre_activation)
        self._post_activation = activations.deserialize(post_activation)
        super().__init__(**kwargs)

    def build(self, input_shapes):
        if not isinstance(input_shapes[0], tuple):
            self.masked = False
            self._input_neurons = input_shapes[-1]
        else:
            self.masked = True
            self._input_neurons = input_shapes[0][-1]

        #self._w = tf.constant(value=1, dtype=K.floatx(), shape=(self._input_neurons, 1))
        # self._q = tf.constant(value=1, dtype=K.floatx(), shape=(1, 1))
        # self._bias = tf.constant(value=0, dtype=K.floatx(), shape=(1, 1))
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

        if self.masked:
            inputs, mask = inputs

        x = self._pre_activation(inputs)

        # Aggregate
        if self.masked:
            x = tf.math.multiply(x, mask, name='apply_mask')
            c = tf.math.add(count, mask, name='count_increment')
        else:
            c = tf.math.add(count, tf.constant([1], dtype=K.floatx()), name='count_increment')
        s = tf.math.add(x, state, name='sum_state')

        #tf.print("JERONE")
        #tf.print(c)
        #tf.print(x)

        # Log transform and apply dense
        x = tf.math.subtract(
            tf.math.multiply(tf.math.log(s, name='x_log_transform'), self._w, name='x_apply_weights'),
            tf.math.multiply(tf.math.log(c, name='c_log_transform'), self._q, name='c_apply_weights'),
            name='dense_linear'
        )
        #tf.print(x)
        x = tf.math.add(
            x,
            self._bias,
            name='dense_bias'
        )
        #tf.print(x)

        x = self._post_activation(x)

        #tf.print(x)
        return x, (s, c)

    def get_config(self):
        return {
            "pre_activation": self._pre_activation,
            "post_activation": self._post_activation,
        }
