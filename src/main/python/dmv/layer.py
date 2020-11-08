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


class DmvCell(Layer):
    agg_methods = ('count', 'sum', 'mean', 'var', 'max', 'sum_log', 'count_log', 'square_sum', 'square_mean', 'custom_sum', 'custom_mean', 'mean_max')

    def __init__(self, aggregations=('mean', ), **kwargs):
        assert all([agg in self.agg_methods for agg in aggregations]), f"All aggregations should be in {self.agg_methods}."
        self._output_aggregations = aggregations
        self._partials = {a: i for i, a in enumerate(self._output_aggregations)}

        self._ensure_presence('mean', 'mean_max')
        self._ensure_presence('max', 'mean_max')
        self._ensure_presence('var', 'std')
        self._ensure_presence('sn', 'var')
        self._ensure_presence('mean', 'sn')
        self._ensure_presence('count', 'mean')
        self._ensure_presence('sum', 'mean')
        self._ensure_presence('count', 'square_mean')
        self._ensure_presence('square_sum', 'square_mean')
        self._ensure_presence('custom_sum', 'custom_mean')

        logger.debug(f'The activated partials are {",".join(self._partials.keys())} of which {",".join(self._output_aggregations)} are returned.')
        super().__init__(**kwargs)

    def _ensure_presence(self, field, given):
        if given in self._partials and field not in self._partials:
            self._partials[field] = len(self._partials)

    def build(self, input_shape):
        if not isinstance(input_shape[0], tuple):
            self.masked = False
            self._input_neurons = input_shape[-1]
        else:
            self.masked = True
            self._input_neurons = input_shape[0][-1]
        self.built = True

    def _tensor_shape(self, agg):
        if agg != 'count' and agg != 'count_log':
            return tf.TensorShape([self._input_neurons])
        else:
            return tf.TensorShape([1])

    @property
    def state_size(self):
        ss = [None] * len(self._partials)
        for p, i in self._partials.items():
            ss[i] = self._tensor_shape(p)
        return ss

    @property
    def output_size(self):
        neurons = sum([self.state_size[self._id(agg)].num_elements() for agg in self._output_aggregations])
        return [tf.TensorShape([neurons])]

    @staticmethod
    def _count(state):
        return tf.math.add(state, tf.constant([1], dtype=K.floatx()), name='dmv_count')

    @staticmethod
    def _sum(state, inputs):
        return tf.math.add(state, inputs, name='dmv_sum')

    @staticmethod
    def _max(state, inputs):
        return tf.math.maximum(state, inputs, name='dmv_max')

    @staticmethod
    def _mean(state_sum, state_count):
        return tf.math.divide(state_sum, state_count, name='dmv_mean')

    @staticmethod
    def _sn(state, prev_mean, cur_mean, inputs):
        return tf.math.add(
            state,
            tf.math.multiply(
                tf.math.subtract(inputs, prev_mean, name='dmv_sn_sub_1'),
                tf.math.subtract(inputs, cur_mean, name='dmv_sn_sub_2'),
                name='dmv_sn_mult'
            ),
            name='dmv_sn_add'
        )

    @staticmethod
    def _var(sn, count):
        return tf.math.divide(sn, count, name='dmv_var')

    @staticmethod
    def _std(var):
        return tf.math.sqrt(tf.maximum(var, 1e-6), name='dmv_std')

    @staticmethod
    def _log(state):
        return tf.math.log(state, name='dmv_log')


    @staticmethod
    def _square(state):
        return tf.math.square(state, name='dmv_square')

    def _id(self, name):
        return self._partials[name]

    def _enabled(self, name):
        return name in self._partials

    def _states_transform(self, states):
        return {name: states[i] for name, i in self._partials.items()}

    def _states_reverse_transform(self, states):
        s = [None] * len(self._partials)
        for name, i in self._partials.items():
            s[i] = states[name]
        return s

    def _states_extract_output(self, states):
        return tf.concat([states[agg] for agg in self._output_aggregations], axis=-1)

    def _log_state(self, name, states):
        tf.print(f"Logging state {name}")
        for name in self._partials.keys():
            tf.print(f'{name}: ', states[name])

    def call(self, inputs, old_states):
        states = self._states_transform(old_states)

        if self._enabled('count'):
            states['count'] = self._count(states['count'])

        if self._enabled('sum'):
            states['sum'] = self._sum(states['sum'], inputs)

        if self._enabled('max'):
            states['max'] = self._max(states['max'], inputs)

        if self._enabled('mean'):
            if self._enabled('sn'):
                states['prev_mean'] = states['mean']
            states['mean'] = self._mean(states['sum'], states['count'])

        if self._enabled('sum_log'):
            states['sum_log'] = self._log(states['sum'])

        if self._enabled('count_log'):
            states['count_log'] = self._log(states['count'])

        if self._enabled('square_sum'):
            states['square_sum'] = self._sum(states['square_sum'], self._square(inputs))

        if self._enabled('square_mean'):
            states['square_mean'] = self._mean(states['square_sum'], states['count'])

        if self._enabled('sn'):
            states['sn'] = self._sn(states['sn'], states['prev_mean'], states['mean'], inputs)

        if self._enabled('var'):
            states['var'] = self._var(states['sn'], states['count'])

        if self._enabled('std'):
            states['std'] = self._std(states['var'])

        if self._enabled('custom_sum'):
            states['custom_sum'] = self._sum(states['custom_sum'], tf.math.multiply(tf.math.substract(tf.constant([1], dtype=K.floatx()), inputs), tf.constant([0], dtype=K.floatx())))

        if self._enabled('custom_mean'):
            states['custom_mean'] = self._mean(states['custom_sum'], states['count'])

        if self._enabled('mean_max'):
            states['mean_max'] = tf.math.divide(tf.math.add(states['mean'], states['max']), tf.constant([2], dtype=K.floatx()))

        return self._states_extract_output(states), self._states_reverse_transform(states)


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
        if not isinstance(input_shapes[0], tuple):
            self.masked = False
            self._input_neurons = input_shapes[-1]
        else:
            self.masked = True
            self._input_neurons = input_shapes[0][-1]

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
