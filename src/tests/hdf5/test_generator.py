import pytest

from rdds.lib.hdf5 import Hd5DataGenerator


def test_hd5_data_generator(hd5_file):
    """
    Test for iterator basic behavior.
    """
    hd5_file_path, data_length = hd5_file
    # GIVEN a HD5 data generator reading from HD5 file
    hd5_data_generator: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path, forever=False)
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
    hd5_data_generator = Hd5DataGenerator(hd5_file_path=hd5_file_path, forever=True)
    # WHEN iterating over data
    data_iter: Hd5DataGenerator = hd5_data_generator()
    # THEN expect data to start over after epoch if 'forever' is True
    for n_epochs in range(0, 2):
        for index in range(0, data_length):
            tensor = data_iter.__next__()
            assert tensor[-1] == index
        assert index == data_length - 1
