from tensorflow.python.keras.saving.saved_model import layer_serialization


class CustomRNNSavedModelSaver(layer_serialization.RNNSavedModelSaver):
    @property
    def object_identifier(self):
        return '_tf_keras_layer'
