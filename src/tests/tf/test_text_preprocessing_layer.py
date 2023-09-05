import tensorflow as tf
import pytest as pt

from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator, TextPreprocessingLayer

@pt.fixture
def string_dataset() -> tf.data.Dataset:
    words = ['A', 'dog', 'barked remotely', 'in', 'the', 'still', 'of', 'the', 'night.',
             'The\ndense mist', 'covered', 'the', 'wast wheat field', 'ahead.']
    return tf.data.Dataset.from_tensor_slices(words, name='textdataset')


def test_preprocessing_layer(string_dataset):

    preprocessing_layer: TextPreprocessingLayer = TextPreprocessingLayer()

    data = preprocessing_layer(tf.constant('foo bar')).numpy()
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('    foo   bar')).numpy()
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('a\nb')).numpy()
    assert data[0][0] == b'a'
    assert data[0][1] == b'b'