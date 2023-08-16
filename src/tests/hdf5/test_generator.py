import pytest as pt
import h5py
import tempfile
import os

from rdds.lib.hdf5 import Hd5DataGenerator
DATA_LENGTH: int = 10

@pt.fixture
def hd5_file_path() -> str:
    file: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
    hd5_file: h5py.File = h5py.File(file.name, 'w')
    group: h5py.Group = hd5_file.create_group('group')
    group.create_dataset('dataset0', dtype=h5py.string_dtype(), shape=(DATA_LENGTH, ))
    group['dataset0'][()] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    group.create_dataset('dataset1', dtype=h5py.string_dtype(), shape=(DATA_LENGTH, ))
    group['dataset1'][()] = ['A', 'B'*2, 'C'*3, 'D'*4, 'E'*5, 'F'*6, 'G'*7, 'H'*8, 'I'*9, 'J'*10]
    group.create_dataset('dataset2', dtype=float, shape=(DATA_LENGTH, ))
    group['dataset2'][()] = list(range(0, DATA_LENGTH))
    hd5_file.flush()
    hd5_file.close()
    yield file.name
    os.remove(file.name)


def test_hd5_data_generator(hd5_file_path):
    """
    Test for iterator basic behavior
    """
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path, forever=False)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    for i, tensor in enumerate(data_iter):
        assert tensor[-1] == i
    # THEN expect the data to be accurately read
    assert tensor == (b'j', b'J'*10, 9)
    assert i == DATA_LENGTH - 1

    # THEN the generator should yield no more data
    with pt.raises(StopIteration):
        data_iter.__next__()


def test_hd5_data_generator_repeat(hd5_file_path):
    """
    Test for repeating data
    """
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path, forever=True)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    # THEN expect data to start over after epoch if 'forever' is True
    for n_epochs in range(0, 2):
        for i in range(0, DATA_LENGTH):
            tensor = data_iter.__next__()
            assert tensor[-1] == i
        assert i == DATA_LENGTH - 1


def test_hd5_data_generator_multiple_tensors(hd5_file_path):
    """
    Test for iterator basic behavior, to return multiple vectors
    """
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            output_tensor_format=[['dataset0', 'dataset1'], ['dataset2']],
                                                            forever=False)
    # WHEN iterating over data
    data_iter = hd5_data_generator()
    for i, nested_tensor in enumerate(data_iter):
        # THEN expect the data to be split according to output_tensor_format
        assert isinstance(nested_tensor, tuple)
        assert len(nested_tensor) == 2
        assert isinstance(nested_tensor[0][0], bytes)
        assert isinstance(nested_tensor[0][1], bytes)
        assert isinstance(nested_tensor[1][0], float)
    assert i == DATA_LENGTH - 1


def test_hd5_data_types(hd5_file_path):
    """
    Test the datatypes returned by hd5 data generator.
    """
    # GIVEN a dataset and generator
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path)
    # WHEN querying the dataset dtypes
    dtypes = hd5_data_generator.data_types
    # THEN expect them to match the data provided
    assert isinstance(dtypes, dict)
    assert dtypes['dataset0'] == bytes
    assert dtypes['dataset1'] == bytes
    assert dtypes['dataset2'] == float
