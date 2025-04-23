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


def test_text_preprocessing_saving(work_dir):
    """
    Test for serializing and saving TextPreprocessingLayer
    """
    # GIVEN a model with TextPreprocessingLayer
    input = tf.keras.Input(shape=(1,), dtype=tf.string, name='strinput')
    text_preprocessing_layer = TextPreprocessingLayer(split_regex='splitme')
    output = text_preprocessing_layer(input)
    model = tf.keras.Model(input, output)
    model.compile()
    # WHEN serializing it for saving
    # THEN expect it to succeed
    tf.keras.models.Model.from_config(model.get_config())
    model.save(filepath=work_dir)
    tf.keras.saving.load_model(work_dir)