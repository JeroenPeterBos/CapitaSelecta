from tensorflow.keras.layers import AbstractRNNCell, RNN, Layer
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.python.framework import tensor_shape


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
    def __init__(self, aggregation_type, **kwargs):
        if aggregation_type == 'max':
            cell = DynamicMultiViewMaxCell(**kwargs)
        elif (aggregation_type == 'mean'):
            cell = DynamicMultiViewMeanCell(**kwargs)
        super().__init__(cell)
    
    def call(self, inputs):
        return super().call(inputs)


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