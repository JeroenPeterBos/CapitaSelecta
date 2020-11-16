from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed, RNN, Layer, Masking, MaxPooling2D
from tensorflow.python.keras import activations
import tensorflow as tf
import tensorflow.keras.backend as K

import logging
logger = logging.getLogger(__name__)


class DynDenseCell(Layer):
    def __init__(self,
                 aggregations=('sum', ),
                 activation=None,
                 count_order=1,
                 **kwargs):
        assert all([aggregation in ['sum', 'max', 'mean', 'var', 'std'] for aggregation in aggregations]), "Agg not supported"

        self._aggregations = aggregations
        self._activation = activation
        self._activation_func = activations.deserialize(activation)
        self._count_order = count_order
        super().__init__(**kwargs)

    def build(self, input_shapes):
        self._input_neurons = input_shapes[-1]

        self._ws = [
            self.add_weight(
                shape=(self._input_neurons,),
                initializer='glorot_uniform',
                name=f'weight_metric_{aggregation}'
            ) for aggregation in self._aggregations
        ]
        self._qs = [
            self.add_weight(
                shape=(self._input_neurons,),
                initializer='glorot_uniform',
                name=f'weight_count_order{i}'
            ) for i in range(1, self._count_order + 1)
        ]
        self._bias = self.add_weight(
            shape=(self._input_neurons,),
            initializer='zeros',
            name='bias'
        )

    @property
    def state_size(self):
        return [tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons]), tf.TensorShape([self._input_neurons])]

    @property
    def output_size(self):
        return [tf.TensorShape([self._input_neurons])]

    def call(self, inputs, states):
        sum_state, max_state, squared_sum_state, count_state = states

        # Aggregate
        new_count = tf.math.add(count_state, tf.constant([1], dtype=K.floatx()), name='count_increment')
        new_sum = tf.math.add(sum_state, inputs, name='sum_state')
        new_max = tf.math.maximum(max_state, inputs, name='max_state')
        new_squared_sum = tf.math.add(squared_sum_state, tf.math.pow(inputs, 2), name='squared_sum_state')
        new_mean = tf.math.divide(new_sum, new_count, name='mean')
        new_var = tf.math.subtract(tf.math.divide(new_squared_sum, new_count), tf.math.pow(new_mean, 2), name='variance')
        new_std = tf.math.sqrt(tf.math.add(new_var, tf.constant(1e-4, shape=(self._input_neurons, ), dtype=K.floatx()), name='standard_deviation'))

        # Multiply metrics to weight and add them together
        x = tf.constant(0, shape=(self._input_neurons, ), dtype=K.floatx())
        for i, aggregation in enumerate(self._aggregations):
            if aggregation == 'sum':
                x = tf.math.add(x, tf.math.multiply(new_sum, self._ws[i]))
            elif aggregation == 'max':
                x = tf.math.add(x, tf.math.multiply(new_max, self._ws[i]))
            elif aggregation == 'mean':
                x = tf.math.add(x, tf.math.multiply(new_mean, self._ws[i]))
            elif aggregation == 'var':
                x = tf.math.add(x, tf.math.multiply(new_var, self._ws[i]))
            elif aggregation == 'std':
                x = tf.math.add(x, tf.math.multiply(new_std, self._ws[i]))
            else:
                raise Exception("This should not happen")

        # Multiply count poly parts by weights and add them together
        y = tf.constant(0, shape=(self._input_neurons, ), dtype=K.floatx())
        for order in range(self._count_order):
            poly_partial = tf.math.multiply(self._qs[order], tf.math.pow(new_count, order + 1))
            y = tf.math.add(y, poly_partial)

        # Add the metrics part and the count poly part together with the biases
        z = tf.math.add(x, y)
        z = tf.math.add(z, self._bias)

        # Apply activation
        if self._activation_func is not None:
            z = self._activation_func(z)

        return z, (new_sum, new_max, new_squared_sum, new_count)

    def get_config(self):
        return {
            'aggregations': self._aggregations,
            'activation': self._activation,
            'count_order': self._count_order
        }


class DynDenseModel(Model):
    def __init__(self, num_classes, input_shape, masked=True, agg_params=None, agg_method='sum', merge_depth=-2, agg_activation='relu', agg_count_order=1, dense_activation='sigmoid'):
        assert merge_depth in [-1, -2]

        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.agg_params = agg_params
        self.agg_method = agg_method
        self.agg_count_order = agg_count_order
        self.merge_depth = merge_depth
        self.agg_activation = agg_activation
        self.dense_activation = dense_activation

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.agg = RNN(DynDenseCell(**self.agg_params))
        self.classify = Dense(num_classes, activation=dense_activation, name='classify')

        logger.info(f"Constructed Dynamic Dense Model with configuration: {self.get_config()}")
        logger.info(f"Constructed Dynamic Dense Cell with configuration: {self.agg.cell.get_config()}")

    def get_config(self):
        return {
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'agg_params': self.agg_params,
            'merge_depth': self.merge_depth,
            'dense_activation': self.dense_activation
        }

    def call(self, inputs, **kwargs):
        x = inputs
        if self.masked:
            x = Masking()(x)

        x = TimeDistributed(self.base)(x)

        if self.merge_depth == -1:
            x = TimeDistributed(self.classify)(x)
            x = self.agg(x)
        elif self.merge_depth == -2:
            x = self.agg(x)
            x = self.classify(x)

        return x


class Mean(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', ),
                'activation': None,
                'count_order': 1
            }
        )

    @staticmethod
    def folder_id():
        return 'dd-mean'


class MeanTanh(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', ),
                'activation': 'tanh',
                'count_order': 1
            }
        )

    @staticmethod
    def folder_id():
        return 'dd-mean-tanh'


class Max(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('max', ),
                'activation': None,
                'count_order': 1
            }
        )


class Std(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('std', ),
                'activation': None,
                'count_order': 1
            }
        )


class Var(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('var', ),
                'activation': None,
                'count_order': 1
            }
        )


class MeanVar(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', 'var', ),
                'activation': None,
                'count_order': 1
            }
        )


class MeanStd(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', 'std', ),
                'activation': None,
                'count_order': 1
            }
        )

    @staticmethod
    def folder_id():
        return 'dd-mean-std'


class MeanStdTanh(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', 'std', ),
                'activation': 'tanh',
                'count_order': 1
            }
        )

    @staticmethod
    def folder_id():
        return 'dd-mean-std-tanh'
        

class MeanMax(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', 'max', ),
                'activation': None,
                'count_order': 1
            }
        )


class MeanStdMax(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('mean', 'std', 'max'),
                'activation': None,
                'count_order': 1
            }
        )


class SumMeanStdMax(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('sum', 'mean', 'std', 'max'),
                'activation': None,
                'count_order': 1
            }
        )


class SumMeanStdMaxTanh(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            agg_params={
                'aggregations': ('sum', 'mean', 'std', 'max'),
                'activation': 'tanh',
                'count_order': 1
            }
        )

# From tensorboard it looks like count_order > 1 does not seem to really help

class LateMean(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            merge_depth=-1,
            dense_activation=None,
            agg_params={
                'aggregations': ('mean',),
                'activation': 'sigmoid',
                'count_order': 1
            }
        )


class LateMeanTanh(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            merge_depth=-1,
            dense_activation='tanh',
            agg_params={
                'aggregations': ('mean',),
                'activation': 'sigmoid',
                'count_order': 1
            }
        )


class LateMeanStd(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            merge_depth=-1,
            dense_activation=None,
            agg_params={
                'aggregations': ('mean', 'std',),
                'activation': 'sigmoid',
                'count_order': 1
            }
        )


class LateMeanStdTanh(DynDenseModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            merge_depth=-1,
            dense_activation='tanh',
            agg_params={
                'aggregations': ('mean', 'std',),
                'activation': 'sigmoid',
                'count_order': 1
            }
        )
