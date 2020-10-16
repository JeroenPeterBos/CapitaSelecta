import os
import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    def __init__(self, root, category=None, train=True, augmentation=None, multi=False, batch_size=8, output_shape=[128, 128, 3], sample_frac=1.0, dtype=tf.float32, cache_imgs=True):
        self._root = root
        self._augmentation = self._aug_params(augmentation)
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
        df['_img_files'] = df.apply(lambda r: [str(self._root / r['folder'] / img_file) for img_file in r['file']], axis=1) 
        if not self._multi:
            df = df.explode('_img_files')
        self.df = df

        # Create the Dataset object
        ds = Dataset \
            .from_generator(self._generate_filenames(), (tf.string, self._dtype), (tf.TensorShape(self._output_shape[:-3]), tf.TensorShape([]))) \
            .cache() \
            .map(self._mapper(self._decode_img()), num_parallel_calls=AUTOTUNE)
        
        if cache_imgs:
            ds = ds.cache()

        if self._augmentation is not None:
            ds = ds.map(self._tf_augmentation_func(), num_parallel_calls=AUTOTUNE)
        
        self._ds = ds

        self.samples = self.df.shape[0]
        self.batches_per_epoch = math.ceil(self.samples / self._batch_size)
    
    @staticmethod
    def _aug_params(aug):
        if aug is not None:
            def f(flip=None):
                rot = aug.get('rotation', 0)
                params = {
                    'theta': np.random.uniform(-rot, rot),
                    'flip_horizontal': (random.randint(0, 1) == 0 if flip is None else flip) if aug.get('horizontal_flip', False) else False
                }
                return params
            return f
        else:
            return None

    def _mapper(self, func):
        if self._multi:
            def f(paths, label):
                return tf.map_fn(func, paths, fn_output_signature=tf.TensorSpec(self._output_shape[-3:], self._dtype)), label
                #tf.stack([func(path, label) for path in tf.unstack(paths)]), label
        else:
            def f(path, label):
                return func(path, label), label
        return f

    def _decode_img(self):
        def f(path, label=None):
            img = tf.io.read_file(path)
            img = tf.io.decode_png(img, channels=self._output_shape[-1])
            img = tf.image.resize_with_pad(img, self._output_shape[-3], self._output_shape[-2])
            img = tf.cast(img, self._dtype)
            img = tf.keras.applications.densenet.preprocess_input(img)
            
            return img
        return f

    def _fill_with_label(self):
        def f(path, label):
            img = self._decode_img()(path, label)
            label_tensor = tf.math.scalar_mul(.1, tf.fill(self._output_shape, label))
            img_tensor = tf.math.scalar_mul(.9, img)
            return tf.reduce_sum([label_tensor, img_tensor], 0)
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
    
    def _py_augmentation_func(self):
        if self._multi:
            def f(image):
                image = image.numpy()
                flip = random.randint(0, 1) == 0
                for i in range(image.shape[0]):
                    image[i] = ImageDataGenerator().apply_transform(x=image[i], transform_parameters=self._augmentation(flip))
                return image
        else:
            def f(image):
                image = image.numpy()
                image = ImageDataGenerator().apply_transform(x=image, transform_parameters=self._augmentation())
                return image
        return f

    def _tf_augmentation_func(self):
        py_func = self._py_augmentation_func()
        def f(images, label):
            images_shape = images.shape
            [images, ] = tf.py_function(py_func, [images], [self._dtype])
            images.set_shape(images_shape)
            return images, label
        return f
    
    def _mura_meta(self):
        df = pd.read_csv(self._root / 'MURA-v1.1' / f'{"train" if self._train else "valid"}_image_paths.csv', header=None, names=['full_path'])

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
    
    def show(self, n=4, figsize=None):
        it = iter(self._ds)

        figm, axs = plt.subplots(4 if self._multi else 1, n, figsize=figsize)
        for i in range(n):
            imgs, label = next(it)

            if self._multi:
                imgs = imgs.numpy().astype(np.float32)

                for j in range(min(4, imgs.shape[0])):
                    img = imgs[j]

                    img = img - img.min()
                    img *= 1.0 / img.max()

                    axs[i, j].imshow(img)
                    axs[i, j].set_title(f'Label: {int(label.numpy())} Range: {("{0:.2f}".format(img.min()), "{0:.2f}".format(img.max()))}')
            else:
                img = imgs.numpy().astype(np.float32)

                img = img - img.min()
                img *= 1.0 / img.max()

                axs[i].imshow(img)
                axs[i].set_title(label.numpy())
        plt.show()
    
    def ds(self):
        return self._ds \
            .shuffle(buffer_size=self.samples) \
            .padded_batch(self._batch_size) \
            .prefetch(buffer_size=AUTOTUNE) \
            .repeat()
