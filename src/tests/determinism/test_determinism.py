import tensorflow as tf
import pytest as pt
import numpy as np
import random as pyrand

from rdds.lib.determinism import enable_determinism, get_seeded_numpy_rng


def test_tensorflow_determinism():
    """
    Test for reproducible initialisation of tensorflow and keras framework.
    """
    # GIVEN some matrixes, generated with a seed
    for seed in range(0, 10):
        # WHEN selecting a random set
        enable_determinism(seed)
        values = tf.random.normal((1, 10000)).numpy()
        enable_determinism(seed)
        new_values = tf.random.normal((1, 10000)).numpy()
        # THEN expect the two matrixes, generated with the same seed, to be identical
        d = np.subtract(new_values, values)
        ds = np.sum(d)
        assert np.isclose(ds, 0, atol=1E-6)


def test_tensorflow_determinism_negative_case():
    """
    Test case for non enabled determinism, i.e. negative case.
    """
    # GIVEN some matrixes, generated with a seed
    # WHEN selecting a random set
    values = tf.random.normal((1, 10000)).numpy()
    new_values = tf.random.normal((1, 10000)).numpy()
    # THEN expect the data to be different tfdeterminism is not enabled
    with pt.raises(AssertionError):
        d = np.subtract(new_values, values)
        ds = np.sum(d)
        assert np.isclose(ds, 0, atol=1E-6)


def test_numpy_generator_determinism():
    """
    Test case for numpy random.Generator determinism
    """
    # GIVEN a seeded Generator
    # WHEN generating values from two identically seeded generators
    values = get_seeded_numpy_rng().random((1, 10000))
    new_values = get_seeded_numpy_rng().random((1, 10000))
    # THEN expect the values to be identical
    d = np.subtract(new_values, values)
    ds = np.sum(d)
    assert np.isclose(ds, 0, atol=1E-6)


def test_python_determinism():
    """
    Test for python determinism
    """
    # GIVEN some matrixes, generated with a seed
    # WHEN selecting a random set
    enable_determinism()
    values = np.array(pyrand.sample(range(10000), k=100))
    enable_determinism()
    new_values = np.array(pyrand.sample(range(10000), k=100))
    d = np.subtract(new_values, values)
    ds = np.sum(d)
    # THEN expect the values to be identical
    assert np.isclose(ds, 0, atol=1E-6)
