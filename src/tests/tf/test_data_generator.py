import pytest as pt
import numpy as np
from typing import *
import tensorflow as tf
from rdds.lib.hdf5 import Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator

from tests.hdf5.test_generator import hd5_file_path, DATA_LENGTH  # Test fixture


@pt.fixture
def endless_hd5_data_generator(hd5_file_path) -> Hd5DataGenerator:
    return Hd5DataGenerator(hd5_file_path=hd5_file_path, forever=True)


@pt.fixture
def output_signature() -> Tuple[Union[tf.TensorSpec, tf.RaggedTensorSpec], ...]:
    """
    Matching data format in hd5_data_generator()
    :return:
    """
    return (tf.TensorSpec((), dtype=tf.string),
            tf.TensorSpec((), dtype=tf.string),
            tf.TensorSpec((), dtype=tf.float32))


def test_hd5_data_generator(endless_hd5_data_generator,
                            output_signature):
    """
    Test for generating data from a tf.data.Dataset class using Hd5DatGenerator as data source.
    """
    # GIVEN a dataset
    hd5_data_set: tf.data.Dataset = \
        get_tf_dataset_from_hd5_data_generator(hd5_data_generator=endless_hd5_data_generator,
                                               output_signature=output_signature)
    hd5_data_set = hd5_data_set.batch(DATA_LENGTH)
    # WHEN reading data from dataset
    for n_epoch in range(0, 5):
        # Select one epoch of data from dataset
        batch: List[Tuple[tf.Tensor]] = list(hd5_data_set.take(1))
        tensors_in_batch: Tuple[tf.Tensor] = batch[0]
        numerical_tensor: tf.Tensor = tensors_in_batch[-1]
        numerical_tensor_as_numpy: np.ndarray = numerical_tensor.numpy()
        # THEN expect data to be presented as expected in every batch
        assert np.isclose(np.sum(np.subtract(numerical_tensor_as_numpy, np.arange(0, DATA_LENGTH))), 0, atol=1E-6)