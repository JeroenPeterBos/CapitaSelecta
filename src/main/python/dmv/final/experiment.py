from argparse import Namespace
from typing import Callable, Tuple
from pathlib import Path
import pandas as pd

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


def load_data(args, meta_file, category, multi):
    train = 'train' in meta_file
    kwargs = {
        'multi': multi,
        'train': train,
        'category': category,
        'batch_size': args.batch_size[int(multi)],
        'output_shape': list(args.img_size),
        'max_imgs': args.max_imgs,
        'shuffle_size': args.shuffle_size if train else 0,
        'cache_imgs': args.cache_data,
    }

    if train:
        kwargs['augmentation'] = {
            'rotation': 30,
            'horizontal_flip': True,
        }

    return DataContainer(args.data, meta_csv=args.data / meta_file, **kwargs)


def train_model(
        args: Namespace,
        model_class: Callable,
        train_dc: DataContainer,
        valid_dc: DataContainer,
        log_folder: Path):
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
    logger.info(f"Instantiating the model with input shape: {instantiation_shape}")
    model: Model = model_class(1, instantiation_shape)

    build_shape =tuple([train_dc._batch_size] + train_dc._output_shape)
    logger.info(f"Building the model with input shape: {build_shape}")
    model.build(build_shape)
    model.compile(
        optimizer=Adam(args.learning_rate, beta_1=0.9, beta_2=0.999),
        metrics=metrics,
        loss=BinaryCrossentropy()
    )

    # Define the callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min', cooldown=2),

        CSVLogger(log_folder / 'log.csv'),
        EpochLogger()
    ]

    if args.tensorboard:
        logger.info(f"Activated tensorboard monitor, logging to: {log_folder}")
        callbacks.append(TensorBoard(log_dir=log_folder, histogram_freq=1))

    logger.info("Activated checkpoints callback")
    callbacks.append(ModelCheckpoint(monitor='val_loss', filepath=log_folder / 'best_weights' / 'cp', save_best_only=True))

    logger.info(f'Starting training of model {model}')
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

    logger.info(f'Finished training, storing weights...')

    model.save_weights(log_folder / 'final_weights' / 'cp')
    model.save(log_folder / 'final_model')

    model.load_weights(log_folder / 'best_weights' / 'cp')
    return model


def evaluate_model(model: Model, test_dc: DataContainer, log_folder: Path):
    score = model.evaluate(test_dc.ds(), steps=test_dc.batches_per_epoch, return_dict=True, verbose=0)

    logger.info(f"The score on the test set is: {score}")

    pd.DataFrame([score]).to_csv(log_folder / 'test_score.csv', index=False)


def predict_model(model: Model, test_dc: DataContainer, log_folder: Path):
    logger.info("Storing the predictions on the test set")
    df = test_dc.df
    df['prediction'] = model.predict(test_dc.ds(), steps=test_dc.batches_per_epoch)

    df.to_csv(log_folder / 'test_predictions.csv', index=False)
