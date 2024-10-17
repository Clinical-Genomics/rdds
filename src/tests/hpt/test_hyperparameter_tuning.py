import pytest
from typing import Tuple, Callable
import tensorflow as tf
import numpy as np
import os

from rdds.lib.hpt import HyperParameters, GridSearchTuner, BayesianTuner, RandomSearchTuner


@pytest.fixture
def mnist_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))


@pytest.fixture
def bootstrap_model(mnist_dataset) -> Tuple[Callable, Callable]:

    (x_train, y_train), (x_test, y_test) = mnist_dataset

    def build_model(hparams: HyperParameters = None):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams.Int('dense', 32, 128, 32), activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams.Float('dropout', 0.2, 0.5, 0.25)),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ])

        model.compile(
          optimizer=hparams.Choice("optimizer", ["adam", "adadelta"]),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'],
        )

        return model

    def fit(model, tuning_callbacks: list):
        history: tf.keras.callbacks.History = model.fit(
          x=x_train,
          y=y_train,
          validation_data=(x_test, y_test),
          epochs=1,
          callbacks=tuning_callbacks
        )
        return history

    return build_model, fit


@pytest.mark.parametrize('Tuner',
                         [GridSearchTuner,
                          BayesianTuner,
                          RandomSearchTuner])
def test_tuners(bootstrap_model, work_dir, Tuner):
    """
    Test for tuners iterating over hyperparameter range and logging to tensorboard directory.
    """
    # GIVEN a model
    build_model, fit = bootstrap_model
    tuner = Tuner(build_fn=build_model,
                  fit_fn=fit,
                  log_dir=work_dir,
                  max_trials=2)

    # WHEN searching for good hyperparameters
    tuner.search()
    # THEN expect the data to be logged to log dir
    # keras_tuner Oracle logging
    assert os.path.exists(os.path.join(work_dir, Tuner.__name__, 'oracle.json'))
    assert os.path.exists(os.path.join(work_dir, Tuner.__name__, 'tuner0.json'))
    for trial_id in ['trial_0', 'trial_1']:
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'trial.json'))
        # Tensorboard logging
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'train'))
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'validation'))


def test_capture_training_errors(bootstrap_model, work_dir):
    """
    Test for erroneous training
    """
    # GIVEN a model with a faulty fit method
    build_model, fit = bootstrap_model

    def faulty_fn(*args, **kwargs):
        raise RuntimeError('Error by design')

    tuner = GridSearchTuner(build_fn=build_model,
                  fit_fn=faulty_fn,
                  log_dir=work_dir,
                  max_trials=2)

    # WHEN searching for hyperparams
    # THEN don't expect a bad training session to fail all runs
    tuner.search()
