import tensorflow as tf
import numpy as np
import random as pyrand

from rdds.lib.logging import get_logger
from .constants import SEED

_LOGGER = get_logger('determinism')


def enable_determinism(seed: int = SEED):
    """
    Set random generator seeds across ecosystem.

    Best practice is to call this method prior to importing any other
    library method, to make sure the seed is set prior to module initialisation.

    Please note that other sources of non-determinism exist, such as
    parallel computations on GPU or in multiprocessing pipelines.
    """
    _enable_python_determinism(seed=seed)
    _enable_numpy_legacy_determinism(seed=seed)
    _enable_tensorflow_determinism(seed=seed)


def _enable_tensorflow_determinism(seed: int):
    """
    Configures tensorflow to run in a deterministic fashion, i.e.
    rerunning a model training will yield identical trained weights.

    Read more in: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
    """
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    _LOGGER.info(f'Tensorflow-Keras determinism enabled with seed {seed}')


def _enable_numpy_generator_determinism(seed: int):
    # https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy
    raise NotImplemented()


def get_seeded_numpy_rng(seed: int = SEED) -> np.random.Generator:
    """
    Convenience function to provide a default seeded numpy Generator.

    This is a workaround for not implementing _enable_numpy_generator_determinism().

    :param seed: The seed
    :return: A seeded numpy Generator
    """
    return np.random.default_rng(seed=seed)


def _enable_numpy_legacy_determinism(seed: int):
    """
    Set seed for numpy random methods in the random module.

    This method does not affect numpy.random.Generator seeding!

    See note from Numpy manual below:

    ---

    This is a convenience, legacy function that exists to support older
    code that uses the singleton RandomState.

    The convenience Functions in numpy.random are still aliases to the
    methods on a single global RandomState instance.

    Best practice is to use a dedicated Generator instance rather than
    the random variate generation methods exposed directly in the random module.

    """
    np.random.seed(seed)
    _LOGGER.info(f'Numpy legacy determinism enabled with seed {seed}')


def _enable_python_determinism(seed: int):
    """
    Set Python random seed.

    Read more in: https://docs.python.org/3/library/random.html#notes-on-reproducibility
    """
    pyrand.seed(seed)
    _LOGGER.info(f'Python determinism enabled with seed {seed}')
