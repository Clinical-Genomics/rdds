import pytest
import h5py
import tempfile
import os
import numpy as np

from rdds.lib.hdf5 import Hd5DataGenerator


def test_hd5_data_generator(hd5_file):
    """
    Test for iterator basic behavior.
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            output_tensor_format=['dataset0', 'dataset1', 'dataset2'],
                                                            forever=False)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    for index, tensor in enumerate(data_iter):
        assert tensor[-1] == index
    # THEN expect the data to be accurately read
    assert tensor == (b'j', b'J'*10, 9)
    assert index == data_length - 1

    # THEN the generator should yield no more data
    with pytest.raises(StopIteration):
        data_iter.__next__()


def test_hd5_data_generator_repeat(hd5_file):
    """
    Test for repeating data.
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                          output_tensor_format=['dataset0', 'dataset1', 'dataset2'],
                                          forever=True)
    # WHEN iterating over data
    data_iter: Hd5DataGenerator = hd5_data_generator()
    # THEN expect data to start over after epoch if 'forever' is True
    for n_epochs in range(0, 2):
        for index in range(0, data_length):
            tensor = data_iter.__next__()
            assert tensor[-1] == index
        assert index == data_length - 1


def test_hd5_data_generator_multiple_tensors(hd5_file):
    """
    Test for iterator basic behavior, to return multiple vectors
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            output_tensor_format=[['dataset0', 'dataset1'], ['dataset2']],
                                                            forever=False)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    for index, nested_tensor in enumerate(data_iter):
        # THEN expect the data to be split according to output_tensor_format
        assert isinstance(nested_tensor, tuple)
        assert len(nested_tensor) == 2
        assert isinstance(nested_tensor[0][0], bytes)
        assert isinstance(nested_tensor[0][1], bytes)
        assert isinstance(nested_tensor[1][0], float)
    assert index == data_length - 1


def test_hd5_with_label(hd5_file):
    """
    Test for yielding labels alongside data.
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            output_tensor_format=[['dataset0', 'dataset1'], ['dataset2']],
                                                            label='label',
                                                            forever=False)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    for i, xy in enumerate(data_iter):
        nested_tensor, label = xy
        # THEN expect the data to be split according to output_tensor_format, and labels are produced
        assert isinstance(nested_tensor, tuple)
        assert len(nested_tensor) == 2
        assert isinstance(nested_tensor[0][0], bytes)
        assert isinstance(nested_tensor[0][1], bytes)
        assert isinstance(nested_tensor[1][0], float)
        assert 0 <= label[0] <= 1
    assert i == data_length - 1

def test_hd5_data_types(hd5_file):
    """
    Test the datatypes returned by hd5 data generator.
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a dataset and generator
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path)
    # WHEN querying the dataset dtypes
    dtypes = hd5_data_generator.data_types
    # THEN expect them to match the data provided
    assert isinstance(dtypes, dict)
    assert dtypes['dataset0'] == bytes
    assert dtypes['dataset1'] == bytes
    assert dtypes['dataset2'] == float
