from argparse import Namespace
from typing import Callable, Tuple
from pathlib import Path

import tensorflow.keras.backend as K

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.python.keras import Model

from tensorflow_addons.metrics import CohenKappa

from dmv.callback import EpochLogger
from dmv.data import DataContainer

import logging
logger = logging.getLogger(__name__)


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
        'max_imgs': max_imgs,
        'shuffle_size': shuffle_size
    }

    train_dc = DataContainer(data_folder, multi=multi, train=True, cache_imgs=cache_imgs, augmentation=train_augmentation, **common)
    valid_dc = DataContainer(data_folder, multi=True, train=False, cache_imgs=True, **common)
    return train_dc, valid_dc


def experiment(
        train_dc: DataContainer,
        valid_dc: DataContainer,
        log_folder: Path,
        model_class: Callable,
        args: Namespace):
    """
    Run an experiment with the given data and model and log the results.

    Args:
        train_dc: Training data container
        valid_dc: Validation data container
        log_folder: The directory to store the results
        model_class: Class to instantiate a model from
        args: The job arguments to run with
    """
    # Define the metrics to track
    metrics = [
        CohenKappa(name='cohen-kappa', num_classes=2),
        TruePositives(name='true-pos'),
        TrueNegatives(name='true-neg'),
        FalsePositives(name='false-pos'),
        FalseNegatives(name='false-neg'),
    ]

    # Define the weights of the classes
    class_weights = {
        0: len(train_dc.df[train_dc.df['label'] == 1]) / len(train_dc.df),
        1: len(train_dc.df[train_dc.df['label'] == 0]) / len(train_dc.df),
    }

    # Construct the model
    instantiation_shape = train_dc._output_shape[-3:]
    logging.info(f"Instantiating the model with input shape: {instantiation_shape}")
    model: Model = model_class(1, instantiation_shape)

    build_shape =tuple([train_dc._batch_size] + train_dc._output_shape)
    logging.info(f"Building the model with input shape: {build_shape}")
    model.build(build_shape)
    model.compile(
        optimizer=Adam(args.learning_rate, beta_1=0.9, beta_2=0.999),
        metrics=metrics,
        loss=BinaryCrossentropy()
    )

    # Define the callbacks
    callbacks = [
        CSVLogger(log_folder / 'log.csv'),
        EpochLogger()
    ]

    if args.tensorboard:
        logging.info(f"Activated tensorboard monitor, logging to: {log_folder}")
        callbacks.append(TensorBoard(log_dir=log_folder, histogram_freq=1))

    if args.checkpoint:
        logging.info("Activated checkpoints callback")
        callbacks.append(ModelCheckpoint(monitor='val_loss', filepath=log_folder / 'saves' / 'checkpoints', save_best_only=True))

    callbacks.extend([
        EarlyStopping(monitor='val_loss', patience=12, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min', cooldown=2),
    ])

    model.fit(
        x=train_dc.ds(),
        steps_per_epoch=train_dc.batches_per_epoch,
        validation_data=valid_dc.ds(),
        validation_steps=valid_dc.batches_per_epoch,
        epochs=args.max_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    model.save_weights(log_folder / 'saves' / 'final_weights')
    model.save(log_folder / 'saves' / 'final_model')
    return model


def evaluate_in_multi_mode(
        model: Model,
        dc: DataContainer,
        log_folder: Path,
        checkpoint: bool):
    model.load_weights(log_folder / 'saves' / 'final_weights')
    final = model.evaluate(dc.ds(), steps=dc.batches_per_epoch, return_dict=True, verbose=0)
    logger.info(f'Final model performance on {dc}: {final}')

    if checkpoint:
        model.load_weights(log_folder / 'saves' / 'checkpoints')
        best = model.evaluate(dc.ds(), steps=dc.batches_per_epoch, return_dict=True, verbose=0)
        logger.info(f'Best model performance on {dc}: {best}')


def validate_saved_model(model, log_folder, valid_dc):
    m = load_model(log_folder / 'saves' / 'final_model', custom_objects={'CohenKappa': CohenKappa})

    logger.info(model.evaluate(valid_dc.ds(), steps=valid_dc.batches_per_epoch, return_dict=True, verbose=0))
    logger.info(m.evaluate(valid_dc.ds(), steps=valid_dc.batches_per_epoch, return_dict=True, verbose=0))

