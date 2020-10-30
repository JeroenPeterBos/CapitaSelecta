from dmv.data import DataContainer
from dmv.layer import Mask
from dmv_tests import constants

import tensorflow as tf


class DataContainerTest(tf.test.TestCase):
    def setUp(self):
        train_augmentation = {
            'rotation': 30,
            'horizontal_flip': True,
        }

        self.dc = DataContainer(
            constants.TRAIN_DATA / 'small_dc_test_files',
            batch_size=2,
            multi=True,
            train=True,
            sample_frac=-1, # disable init shuffle
            shuffle_size=0, # disable epoch shuffle
            augmentation=train_augmentation
        )

        self.imgs_counts = [
            [3, 4],
            [1, 2],
            [4, 3]
        ]

    def test_init(self):
        self.assertEqual(len(self.dc), 6)

    def test_pre_batch_loading(self):
        ds = iter(self.dc._ds)
        flat = [x for y in self.imgs_counts for x in y]
        for i, c in zip(range(len(self.dc)), flat):
            elem = next(ds)

            self.assertEqual(elem[0].shape[0], c)

    def test_loading(self):
        ds = iter(self.dc.ds())
        for i in range(self.dc.batches_per_epoch):
            elem = next(ds)

            self.assertEqual(elem[0].shape[0], 2)
            self.assertEqual(elem[0].shape[1], max(self.imgs_counts[i]))
            self.assertEqual(elem[0].shape[2], 128)
            self.assertEqual(elem[0].shape[3], 128)
            self.assertEqual(elem[0].shape[4], 3)

            self.assertEqual(elem[1].shape[0], 2)

    def test_padded(self):
        ds = iter(self.dc.ds())

        for _, ic in zip(range(self.dc.batches_per_epoch), self.imgs_counts):
            imgs = next(ds)[0]
            count = imgs.shape[1]

            for i in (0, 1):
                actual = ic[i]

                for j in range(0, actual):
                    self.assertNotAllEqual(imgs[i, j, :, :, :], tf.zeros([128, 128, 3]))

                for j in range(actual, count):
                    self.assertAllEqual(imgs[i, j, :, :, :], tf.zeros([128, 128, 3]))

    def test_mask(self):
        masker = Mask()
        ds = iter(self.dc.ds())

        for _, ic in zip(range(self.dc.batches_per_epoch), self.imgs_counts):
            imgs = next(ds)[0]

            for time_dist in range(imgs.shape[1]):
                mask = masker(imgs[:, time_dist, :, :, :])

                self.assertAllEqual(mask, tf.constant([[ic[0] > time_dist], [ic[1] > time_dist]], dtype=tf.float32))
