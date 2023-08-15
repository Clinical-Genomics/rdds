import tensorflow as tf

from rdds.lib.tf import TextPreprocessingLayer


def test_preprocessing_layer():
    """
    Test text preprocessing layer splitting functionality.
    """
    # GIVEN a text preprocessing layer
    preprocessing_layer = TextPreprocessingLayer()
    # WHEN processing the data
    data = preprocessing_layer(tf.constant('foo bar')).numpy()
    # THEN expect data to be appropriately split
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('    foo   bar')).numpy()
    assert data[0][0] == b'foo'
    assert data[0][1] == b'bar'

    data = preprocessing_layer(tf.constant('a\nb')).numpy()
    assert data[0][0] == b'a'
    assert data[0][1] == b'b'


def test_preprocessing_layer_multi_tensor():
    """
    Test for providing multiple tensors as input to TextPreprocessingLayer.
    """
    # GIVEN a preprocessing layer
    preprocessing_layer: TextPreprocessingLayer = TextPreprocessingLayer()
    # WHEN providing multiple tensors as input
    t0, t1 = preprocessing_layer(tf.constant(['foo bar']), tf.constant('lol cat'))
    # THEN expect individual tensors back, each processed
    assert t0.shape == (1, None)
    assert t0.numpy()[0][0] == b'foo'
    assert t1.shape == (1, None)
    assert t1.numpy()[0][1] == b'cat'
