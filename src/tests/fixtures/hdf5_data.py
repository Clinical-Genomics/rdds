import pytest
import h5py
from typing import Tuple
import tempfile
import os
import numpy as np

from rdds.lib.determinism import SEED


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
    group.create_dataset('label', dtype=float, shape=(data_length, ))
    group['label'][()] = np.array(list(range(0, data_length))) / data_length
    hd5_file.flush()
    hd5_file.close()
    yield file.name, data_length
    os.remove(file.name)

@pytest.fixture
def hd5_file_path_with_nans(hd5_file):
    hd5_file_path, data_length = hd5_file
    hd5_file: h5py.File = h5py.File(hd5_file_path, 'r+')
    group = hd5_file['group']
    group.create_dataset('string_null', shape=(data_length, ), dtype=h5py.string_dtype())
    group.create_dataset('float_null', shape=(data_length,), dtype=float)
    # Null bytestrings are empty b''
    group['float_null'][()] = [None] * data_length  # Null floats are NaN
    print(group['string_null'][:])
    print(group['float_null'][:])
    hd5_file.flush()
    hd5_file.close()
    yield hd5_file_path

@pytest.fixture
def hd5_file_path_with_categorical_labels(hd5_file) -> str:
    hd5_file_path, data_length = hd5_file
    hd5_file: h5py.File = h5py.File(hd5_file_path, 'r+')
    group = hd5_file['group']
    categorical_labels = np.random.default_rng(seed=SEED).integers(low=0, high=2, size=data_length)
    group['label'][()] = categorical_labels[:]
    print('LABEL', group['label'][:])
    hd5_file.flush()
    hd5_file.close()
    return hd5_file_path
