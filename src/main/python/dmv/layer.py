from tensorflow.keras.layers import AbstractRNNCell, RNN
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
import tensorflow.keras.backend as K


class DynamicMultiViewRNNCell(AbstractRNNCell):
    def build(self, input_shape):
        self.units = input_shape[-1]
        self.built = True
        
    @property
    def state_size(self):
        return self.units
    
    def call(self, inputs, states):
        output = K.maximum(inputs, states[0])
        return output, output


class DynamicMultiViewRNN(RNN):
    def __init__(self, **kwargs):
        cell = DynamicMultiViewRNNCell(**kwargs)
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