from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras import Model
import tensorflow as tf

class MultiEvalModel(Model):    
    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        x_mask = tf.expand_dims(tf.cast(
                tf.math.greater(
                    tf.math.count_nonzero(x, axis=[2, 3, 4]), 
                    tf.constant([0], dtype=tf.int64)
                ),
                tf.keras.backend.floatx()), axis=-1)

        x = tf.transpose(x, perm=[1, 0, 2, 3, 4])
        y_preds = tf.map_fn(fn=lambda t: self(t, training=False), elems=x)
        y_preds = tf.transpose(y_preds, perm=[1, 0, 2])

        y_pred = tf.math.divide(
            tf.math.reduce_sum(tf.math.multiply(y_preds, x_mask), axis=1), 
            tf.math.reduce_sum(x_mask, axis=1)
        )

        # Updates stateful loss metrics.
        self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}