import pytest
import tempfile
import shutil
from typing import List
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
