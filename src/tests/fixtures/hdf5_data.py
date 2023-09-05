import pytest
import h5py
from typing import Tuple
import tempfile
import os


@pytest.fixture
def hd5_file() -> Tuple[str, int]:
    file: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
    hd5_file = h5py.File(file.name, 'w')
    group: h5py.Group = hd5_file.create_group('group')
    data_length: int = 10
    group.create_dataset('dataset0', dtype=h5py.string_dtype(), shape=(data_length, ))
    group['dataset0'][()] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    group.create_dataset('dataset1', dtype=h5py.string_dtype(), shape=(data_length, ))
    group['dataset1'][()] = ['A', 'B'*2, 'C'*3, 'D'*4, 'E'*5, 'F'*6, 'G'*7, 'H'*8, 'I'*9, 'J'*10]
    group.create_dataset('dataset2', dtype=float, shape=(data_length, ))
    group['dataset2'][()] = list(range(0, data_length))
    hd5_file.flush()
    hd5_file.close()
    yield file.name, data_length
    os.remove(file.name)
