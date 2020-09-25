import os
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE


def _benchmark(dataset, num_epochs=1, sleep=0.0):
    start_time = time.perf_counter()
    counter = 0
    for epoch_num in range(num_epochs):
        for sample in dataset:
            counter += 1
            if sleep > 0:
                time.sleep(sleep)
    tf.print("Execution time:", (time.perf_counter() - start_time) / num_epochs)
    tf.print("Loaded batches:", counter // num_epochs)


class DataContainer:
    def __init__(self, root, category=None, train=True, augmentation_func=None, multi=False, batch_size=8, output_shape=[128, 128, 3], sample_frac=1.0, dtype=tf.float32):
        self._root = root
        self._augmentation_func = augmentation_func
        self._multi = multi
        self._batch_size = batch_size
        self._output_shape = [None] + output_shape if self._multi else output_shape
        self._train = train
        self._dtype = dtype

        # Load the Meta Dataframe
        df = self._mura_meta()
        if category is not None:
            df = df[df['location'] == category]
        df = df.sample(frac=sample_frac)
        df['_img_files'] = df.apply(lambda r: [os.path.join(self._root, r['folder'], img_file) for img_file in r['file']], axis=1) 
        if not self._multi:
            df = df.explode('_img_files')
        self.df = df

        # Create the Dataset object
        ds = Dataset \
            .from_generator(self._generate_filenames(), (tf.string, tf.int16), (tf.TensorShape(self._output_shape[:-3]), tf.TensorShape([]))) \
            .map(self._decode_multi_sample() if self._multi else self._decode_single_sample(), num_parallel_calls=AUTOTUNE) \
            .cache()
        
        if self._augmentation_func is not None:
            ds = ds.map(self._tf_augmentation_func(), num_parallel_calls=AUTOTUNE)
        
        self._ds = ds

        self.samples = self.df.shape[0]
        self.batches_per_epoch = math.ceil(self.samples / self._batch_size)
    
    def _decode_img(self):
        def f(path):
            img = tf.io.read_file(path)
            img = tf.io.decode_png(img, channels=self._output_shape[-1])
            img = tf.image.resize_with_crop_or_pad(img, self._output_shape[-3], self._output_shape[-2])
            img = tf.image.convert_image_dtype(img, dtype=self._dtype)
            
            return img
        return f

    def _decode_single_sample(self):
        def f(path, label):
            return self._decode_img()(path), label
        return f


    def _decode_multi_sample(self):
        def f(paths, label):
            return tf.map_fn(self._decode_img(), paths, fn_output_signature=tf.TensorSpec(self._output_shape[-3:], dtype=self._dtype)), label
        return f
    
    def _generate_filenames(self):
        if self._multi:
            def f():
                for _, row in self.df.iterrows():
                    yield(tf.stack(row['_img_files']), row['label'])
        else:
            def f():
                for _, row in self.df.iterrows():
                    yield(row['_img_files'], row['label'])
        return f
    
    def _tf_augmentation_func(self):
        def f(images, label):
            images_shape = images.shape
            [images, ] = tf.py_function(self._augmentation_func, [images], [self._dtype])
            images.set_shape(images_shape)
            return images, label
        return f
    
    def _mura_meta(self):
        df = pd.read_csv(os.path.join(self._root, 'MURA-v1.1', f'{"train" if self._train else "valid"}_image_paths.csv'), header=None, names=['full_path'])

        components = df['full_path'].str.split('/')
        df['folder'] = components.str[:-1].str.join('/')
        df['file'] = components.str[-1]

        df = df.groupby('folder').agg({'file': lambda x: list(x)}).reset_index()

        components = df['folder'].str.split('/')
        df['location'] = components.str[2].str.split('XR_').str[1].str.title()
        df['patient'] = components.str[3].str.replace('patient', '')
        df['session'] = components.str[4].str.split('_').str[0].str.replace('study', '')
        df['study'] = df['patient'] + '-' + df['session']

        df['label'] = ((components.str[4]).str.contains('positive')).astype(int)
        df['index'] = df.index
        return df
    
    def show(self, n=4):
        it = iter(self._ds)

        figm, axs = plt.subplots(1, n)
        for i in range(n):
            imgs, label = next(it)
            print(imgs, label)

            if self._multi:
                imgs = imgs.numpy().astype(np.float32)
                axs[i].imshow(imgs[0])
                axs[i].set_title(label.numpy())
            else:
                imgs = imgs.numpy().astype(np.float32)
                axs[i].imshow(imgs)
                axs[i].set_title(label.numpy())
        plt.show()
    
    def ds(self):
        return self._ds \
            .shuffle(buffer_size=self.samples) \
            .padded_batch(self._batch_size) \
            .prefetch(buffer_size=AUTOTUNE) \
            .repeat()
