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


class MultiViewDecisionLevelFusionModel(Model):
    def __init__(self, num_classes, input_shape, aggregation_type, masked, aggregation_layer=-1, post_activation=False):
        assert aggregation_layer in [-1, -2], "Aggregation layer must be either the last or one but last layer."

        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.aggregation_type = aggregation_type
        self.aggregation_layer = aggregation_layer
        self.post_activation = post_activation

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.classify = Dense(num_classes, activation='sigmoid' if not post_activation or aggregation_layer == -2 else None, name='classify')
        self.agg = RNN(cell=self._translate_cell(aggregation_type)())

    @staticmethod
    def _translate_cell(cell):
        return {
            'mean': DynamicMultiViewMeanCell,
        }[cell]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'aggregation_type': self.aggregation_type,
            'post_activation': self.post_activation
        })
        logger.debug(f'dlf: {config}')
        return config

    def call(self, inputs, **kwargs):
        x = inputs
        if self.masked:
            x = Masking()(x)

        x = TimeDistributed(self.base)(x)

        if self.aggregation_layer == -1:
            x = TimeDistributed(self.classify)(x)

            x = self.agg(x)
            if self.post_activation:
                x = tf.keras.activations.deserialize('sigmoid')(x)
        elif self.aggregation_layer == -2:
            x = self.agg(x)
            if self.post_activation:
                x = tf.keras.activations.deserialize('sigmoid')(x)

            x = self.classify(x)

        return x


class Mean(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=True)

    @staticmethod
    def folder_id():
        return "fusion-mean"


class MeanPost(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=True, post_activation=True)

    @staticmethod
    def folder_id():
        return "fusion-mean-post"


class MeanEmbedding(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=True, aggregation_layer=-2)

    @staticmethod
    def folder_id():
        return "fusion-mean-emb"


class MeanPostEmbedding(MultiViewDecisionLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(num_classes=num_classes, input_shape=input_shape, aggregation_type='mean', masked=True, aggregation_layer=-2, post_activation=True)

    @staticmethod
    def folder_id():
        return "fusion-mean-post-emb"
