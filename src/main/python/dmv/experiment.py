from typing import Callable, Tuple
from pathlib import Path

from tensorflow.keras.metrics import Accuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard

from tensorflow_addons.metrics import CohenKappa

from dmv.data import DataContainer


def load_data(
        data_folder: Path,
        multi: bool,
        img_shape: tuple = (320, 320, 3),
        batch_size: Tuple[int, int] = (8, 4),
        category: str = None,
        cache_imgs: bool = True,
        max_imgs: int = 8,
        shuffle_size: int = -1):
    """
    Load the data containers.

    Args:
        data_folder: The root directory of the MURA dataset
        multi: Train data in multi view or single view format
        img_shape: The image dimensions to load the data in
        batch_size: A pair for (single, multi) batch_sizes or simply a fixed batch size
        category: Select a subset of all the data by category
        cache_imgs: Store the images in memory before augmentation
        max_imgs: The maximum number of images in a study for it to be included in the dataset
        shuffle_size: The number of samples to load when shuffling.

    Returns:
        The train data container and the validation data container
    """
    train_augmentation = {
        'rotation': 30,
        'horizontal_flip': True,
    }

    common = {
        'category': category,
        'batch_size': batch_size[int(multi)],
        'output_shape': list(img_shape),
        'cache_imgs': cache_imgs,
        'max_imgs': max_imgs,
        'shuffle_size': shuffle_size
    }

    train_dc = DataContainer(data_folder, multi=multi, train=True, augmentation=train_augmentation, **common)
    valid_dc = DataContainer(data_folder, multi=True, train=False, **common)
    return train_dc, valid_dc


def experiment(
        train_dc: DataContainer,
        valid_dc: DataContainer,
        log_folder: Path,
        model_class: Callable,
        max_epochs=100):
    """
    Run an experiment with the given data and model and log the results.

    Args:
        train_dc: Training data container
        valid_dc: Validation data container
        log_folder: The directory to store the results
        model_class: Class to instantiate a model from
        max_epochs: Maximum number of epoch to train for
    """
    # Define the metrics to track
    metrics = [
        CohenKappa(name='cohen-kappa', num_classes=2),
        TruePositives(name='true-pos'),
        TrueNegatives(name='true-neg'),
        FalsePositives(name='false-pos'),
        FalseNegatives(name='false-neg'),
        Accuracy()
    ]

    # Define the weights of the classes
    class_weights = {
        0: len(train_dc.df[train_dc.df['label'] == 1]) / len(train_dc.df),
        1: len(train_dc.df[train_dc.df['label'] == 0]) / len(train_dc.df),
    }

    # Construct the model
    model = model_class(1, train_dc._output_shape[-3:])

    model.build(tuple([train_dc._batch_size] + train_dc._output_shape))
    model.compile(
        optimizer=Adam(0.0001, beta_1=0.9, beta_2=0.999),
        metrics=metrics,
        loss=BinaryCrossentropy()
    )

    # Define the callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min'),
        ModelCheckpoint(monitor='val_loss', filepath=log_folder / 'checkpoints' / 'checkpoint', save_best_only=True),

        TensorBoard(log_dir=log_folder, histogram_freq=1),
        CSVLogger(log_folder / 'log.csv')
    ]

    model.fit(
        x=train_dc.ds(),
        steps_per_epoch=train_dc.batches_per_epoch,
        validation_data=valid_dc.ds(),
        validation_steps=valid_dc.batches_per_epoch,
        epochs=max_epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )
