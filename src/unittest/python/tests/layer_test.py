from dmv.layer import Mask, DynamicMultiViewRNN

import tensorflow as tf
from unittest import TestCase


class MaskTest(tf.test.TestCase):
    img_shape = [320, 320, 3]

    def test_mask(self):
        mask = [0, 1, 0, 1, 0, 1, 0, 1]
        mask_tensor = tf.expand_dims(tf.constant(mask, dtype=tf.float32), -1)
        img_tensor = tf.stack([tf.zeros(self.img_shape) if mv == 0 else tf.ones(self.img_shape) for mv in mask])
        masker = Mask()

        mask_res = masker(img_tensor)

        self.assertAllEqual(mask_tensor, mask_res)


class DynamicMultiViewRNNTest(tf.test.TestCase):
    def setUp(self):
        self.data = [
            [
                [
                    [2],
                    [3],
                    [4]
                ]
            ],
            [
                [
                    [7],
                    [6],
                    [5]
                ]
            ]
        ]
        self.mask = [
            [
                [
                    [1],
                    [1],
                    [0]
                ]
            ],
            [
                [
                    [0],
                    [1],
                    [0]
                ]
            ]
        ]

    def test_masked_mean(self):
        rnn = DynamicMultiViewRNN(aggregation_type='mean')
        data = tf.constant(self.data)
        mask = tf.constant(self.mask)

        for i in range(data.shape[1]):
            res = rnn((data[:, i, :], mask[:, i, :]))

        self.assertAllEqual(tf.constant([[2.5], [6]]), res)

    def test_unmasked_mean(self):
        rnn = DynamicMultiViewRNN(aggregation_type='mean')
        data = tf.constant(self.data, dtype=tf.float32)

        for i in range(data.shape[1]):
            res = rnn(data[:, i, :])

        self.assertAllEqual(tf.constant([[3], [6]]), res)

    def test_masked_max(self):
        rnn = DynamicMultiViewRNN(aggregation_type='max')
        data = tf.constant(self.data)
        mask = tf.constant(self.mask)

        for i in range(data.shape[1]):
            res = rnn((data[:, i, :], mask[:, i, :]))

        self.assertAllEqual(tf.constant([[3], [6]]), res)

    def test_unmasked_max(self):
        rnn = DynamicMultiViewRNN(aggregation_type='max')
        data = tf.constant(self.data)

        for i in range(data.shape[1]):
            res = rnn(data[:, i, :])

        self.assertAllEqual(tf.constant([[4], [7]]), res)
