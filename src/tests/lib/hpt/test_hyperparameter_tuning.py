import gc
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

    def build_model(hparams: HyperParameters = None,
                    trial_work_dir = None):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams.Int('dense', 32, 128, 32), activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams.Float('dropout', 0.2, 0.5, 0.25)),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
        ])

        model.compile(
          optimizer=hparams.Choice("optimizer", ["adam", "adadelta"]),
          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
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
    assert os.path.exists(os.path.join(work_dir, Tuner.__name__, 'hyperparameter-search-space-summary.txt'))
    assert os.path.exists(os.path.join(work_dir, Tuner.__name__, 'tuner0.json'))
    for trial_id in ['trial_0', 'trial_1']:
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'trial.json'))
        # Tensorboard logging
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'train'))
        assert os.path.exists(os.path.join(work_dir, Tuner.__name__, trial_id, 'validation'))


@pytest.mark.parametrize('Tuner',
                         [GridSearchTuner,
                          BayesianTuner,
                          RandomSearchTuner])
@pytest.mark.parametrize('seed',
                         [None,
                          0,
                          1,
                          2])
def test_tuner_seed(work_dir, Tuner, seed):
    """
    Test for setting seed.
    """
    # GIVEN a tuner
    # WHEN setting the seed
    tuner = Tuner(build_fn=None,
                  fit_fn=None,
                  log_dir=work_dir,
                  seed=seed)
    # THEN expect it to be properly set in the oracle
    if seed is None or seed == 0:
        assert tuner.oracle.seed >= 0
    else:
        assert tuner.oracle.seed == seed


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


def test_logdir(work_dir):
    """
    Test for checking trial log dir generation
    """

    class DummyHistory:
        @property
        def history(self):
            return {'val_loss': [-1.0]}

    def check_logdir_fn(hparams: HyperParameters, trial_work_dir: str):
        assert trial_work_dir == f'{work_dir}/{GridSearchTuner.__name__}/trial_0'
        hparams.Int('dummyInt', 0, 1)

    # GIVEN a tuner
    # WHEN running a trial
    # THEN make sure a trial work dir is set
    tuner = GridSearchTuner(build_fn=check_logdir_fn,
                            fit_fn=lambda *args, **kwargs: DummyHistory(),
                            log_dir=work_dir,
                            max_trials=1)
    tuner.search()


def test_data_leakage(work_dir):
    """
    Test for checking no data leaks (model or tf.data.Data instances) during HPT tuning (across iterations).
    """

    # GIVEN a hyperparameter training run

    n_trials = 100
    step = 1.0 / n_trials

    # WHEN running hyperparameter trials
    # THEN expect no objects to be lingering from the previous run

    class DummyModel:
        pass

    def build_model(hparams: HyperParameters,  trial_work_dir: str):
        # hparams is a context manager, that keeps track of objects created in it's scope.
        v = hparams.Float('value', min_value=0, max_value=1, default=1, step=step)

        # Create some dummy data
        N_GIGABYTES = 1
        GIGABYTE = 1024 * 1024 * 1024
        data = np.ones(shape=(GIGABYTE, N_GIGABYTES), dtype='int8')
        data_from_tensors = tf.data.Dataset.from_tensors(data)
        data = data_from_tensors.cache()

        # Create a model and store some data, hparams in the object
        model = DummyModel()
        model.data = data
        model.v = float(v)  # Keep reference in hparams

        # Query garbage collector and check for stale objects from last iteration
        # If there are stale objects, then fail the test
        for ref in gc.get_objects():
            if isinstance(ref, tf.data.Dataset):
                assert hex(id(ref)) in [hex(id(data_from_tensors)), hex(id(data))], f'Stale dataset is not gc\'ed: {ref.__class__} {hex(id(ref))}'
            if isinstance(ref, DummyModel):
                assert hex(id(ref)) == hex(id(model)), f'Stale model is not gc\'ed {ref.__class__} {hex(id(ref))}'

        return model

    def fit(model, tuning_callbacks: list):
        # A dummy test fit function

        class DummyHistory:

            @property
            def history(self):
                return {
                    'val_loss': [model.v]
                }
        return DummyHistory()

    tuner = GridSearchTuner(build_fn=build_model,
                            fit_fn=fit,
                            log_dir=work_dir,
                            max_trials=4)
    tuner.search()
