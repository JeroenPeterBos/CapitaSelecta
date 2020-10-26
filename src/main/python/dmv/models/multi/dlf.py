from tensorflow.python.keras import Model
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.layers import Dense, TimeDistributed

from dmv.layer import Mask, DynamicMultiViewRNN


class MultiViewDecisionLevelFusionModel(Model):
    def __init__(self, num_classes, input_shape, aggregation_type, masked):
        self.masked = masked

        super().__init__()
        self.base = DenseNet121(include_top=False, input_shape=input_shape, pooling='avg')
        for index, layer in enumerate(self.base.layers):
            layer.trainable = True

        self.classify = Dense(num_classes, activation='sigmoid', name='classify')
        self.agg = DynamicMultiViewRNN(aggregation_type=aggregation_type)

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
