from typing import List

from tensorflow.keras.applications import DenseNet121
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, TimeDistributed

from dmv.layer import DynamicMultiViewRNN, Mask


class MultiViewFeatureLevelFusionModel(Model):
    def __init__(self, num_classes, input_shape, aggregation_type, masked):
        super().__init__()
        self.num_classes = num_classes
        self.my_input_shape = input_shape
        self.masked = masked
        self.aggregation_type = aggregation_type

        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.agg = DynamicMultiViewRNN(aggregation_type=aggregation_type)

        self.classify = Dense(num_classes, activation='sigmoid', name='classify')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.my_input_shape,
            'masked': self.masked,
            'aggregation_type': self.aggregation_type
        })
        return config

    def call(self, inputs, **kwargs):
        x = TimeDistributed(self.base)(inputs)

        if self.masked:
            y = TimeDistributed(Mask())(inputs)
            x = self.agg((x, y))
        else:
            x = self.agg(x)

        x = self.classify(x)
        return x


class Max(MultiViewFeatureLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            aggregation_type='max',
            masked=False
        )


class MaxMasked(MultiViewFeatureLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            aggregation_type='max',
            masked=True
        )


class Mean(MultiViewFeatureLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            aggregation_type='mean',
            masked=False
        )


class MeanMasked(MultiViewFeatureLevelFusionModel):
    def __init__(self, num_classes, input_shape):
        super().__init__(
            num_classes=num_classes,
            input_shape=input_shape,
            aggregation_type='mean',
            masked=True
        )