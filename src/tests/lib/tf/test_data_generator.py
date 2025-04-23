import pytest
import numpy as np
from typing import Tuple, Union, List
import tensorflow as tf
from rdds.lib.hdf5 import Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator


@pytest.fixture
def hd5_data_generator(hd5_file) -> Tuple[Hd5DataGenerator, int]:
    hd5_file_path, data_length = hd5_file
    return Hd5DataGenerator(hd5_file_path=hd5_file_path,
                            output_tensor_format=['dataset0', 'dataset1', 'dataset2'],
                            label='label'), data_length

@pytest.fixture
def output_signature() -> Tuple[Union[tf.TensorSpec, tf.RaggedTensorSpec], ...]:
    """
    Matching data format in hd5_data_generator()
    :return:
    """
    return ((tf.TensorSpec((), dtype=tf.string),  # string 0
            tf.TensorSpec((), dtype=tf.string),  # string 1
            tf.TensorSpec((), dtype=tf.float32)),  # numerical 0

            (tf.TensorSpec((2, ), dtype=tf.float32), ))  # label 2D


def test_hd5_data_generator(hd5_data_generator,
                            output_signature):
    """
    Test for generating data from a tf.data.Dataset class using Hd5DatGenerator as data source.
    """
    data_generator, data_length = hd5_data_generator
    # GIVEN a dataset
    hd5_data_set: tf.data.Dataset = \
        get_tf_dataset_from_hd5_data_generator(hd5_data_generator=data_generator,
                                               output_signature=output_signature)
    hd5_data_set = hd5_data_set.repeat(-1)  # endless dataset
    hd5_data_set = hd5_data_set.batch(data_length)
    # WHEN reading data from dataset
    for n_epoch in range(0, 5):
        # Select one epoch of data from dataset
        batch: List[Tuple[tf.Tensor]] = list(hd5_data_set.take(1))
        xy: Tuple[tf.Tensor] = batch[0]
        data_tensors = xy[0]
        label_tensors = xy[1]
        numerical_tensor: tf.Tensor = data_tensors[-1]  # label tensor is the last tensor
        numerical_tensor_as_numpy: np.ndarray = numerical_tensor.numpy()
        # THEN expect data to be presented as expected in every batch
        assert np.isclose(np.sum(np.subtract(numerical_tensor_as_numpy, np.arange(0, data_length))), 0, atol=1E-6)
        assert np.max(label_tensors) <= 1
        assert np.min(label_tensors) >= 0


def test_dataset_repeat(hd5_data_generator, output_signature):
    """
    Test that hd5 data generator, TF data.Dataset can create repeatable datasets, which is
    the base for lots of other functionality in tf.data.Dataset
    """
    data_generator, data_length = hd5_data_generator
    # GIVEN a hd5 data generator
    hd5_data_set: tf.data.Dataset = \
        get_tf_dataset_from_hd5_data_generator(hd5_data_generator=data_generator,
                                               output_signature=output_signature)

    # WHEN repeating the dataset twice
    hd5_data_set = hd5_data_set.repeat(count=2)

    # THEN expect the data to be repeated
    assert len(list(hd5_data_set)) == data_length * 2
