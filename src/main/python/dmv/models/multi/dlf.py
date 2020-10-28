from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed

from dmv.layer import Mask, DynamicMultiViewRNN


import logging
logger = logging.getLogger(__name__)


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
        self.agg = DynamicMultiViewRNN(aggregation_type=aggregation_type)

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
        x = TimeDistributed(self.base)(inputs)
        x = TimeDistributed(self.classify)(x)

        if self.masked:
            y = TimeDistributed(Mask())(inputs)
            x = self.agg((x, y))
        else:
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
