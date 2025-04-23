import numpy as np
import pytest
import tempfile
import shutil
from typing import List
import os
import tensorflow as tf


@pytest.fixture
def work_dir() -> str:
    """
    Return a temporary working directory.
    :return: A path
    """
    work_dir = tempfile.mkdtemp(dir='/tmp')
    yield work_dir
    shutil.rmtree(work_dir)


@pytest.fixture
def word_dataset() -> tf.data.Dataset:
    """
    Return a tensorflow.data.Dataset containing a sentence.
    :return: tf.data.Dataset
    """
    words: List[str] = ['SomeSPLITMEnicely', 'formatted', 'sentence']
    return tf.data.Dataset.from_tensor_slices(words)


@pytest.fixture()
def feature_columns_dataset() -> tf.data.Dataset:
    # Two textual features with varying length contents depending on data row
    # 10 rows of data
    # Dataset yields one tensor with shape [BATCH_SIZE, N_FEATURES] = [4, 2]
    x = [
        ['first feature', 'next_feature'],
        ['bar', 'that contains some new data'],
        ['cat', 'with a'],
        ['jazz sense', '4th data row!'],
        ['lots of glitter', ''],
        ['cake and champagne', 'that\'s'],
        ['pretty nice', 'coming'],
        ['to an end', ''],
        ['this is', 'the'],
        ['', 'end.']
    ]
    x = np.asarray(x)
    x = tf.constant(x)
    x = tf.RaggedTensor.from_tensor(x)
    dataset = tf.data.Dataset.from_tensor_slices(x)
    dataset = dataset.batch(4)
    return dataset


@pytest.fixture
def vocabulary_file():
    file_path = '/tmp/vocab.txt'
    with open(file_path, 'w') as vocabulary_file:
        for word in ['foo', 'bar']:
            print(word, file=vocabulary_file)
    yield file_path
    os.remove(file_path)
