import tensorflow as tf

from rdds.lib.tf import TextPreprocessingLayer


def test_preprocessing_layer():

    preprocessing_layer = TextPreprocessingLayer()

    data = preprocessing_layer(tf.constant('foo bar')).numpy()
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('    foo   bar')).numpy()
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('a\nb')).numpy()
    assert data[0][0] == b'a'
    assert data[0][1] == b'b'