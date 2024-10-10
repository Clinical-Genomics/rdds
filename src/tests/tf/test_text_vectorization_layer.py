import tensorflow as tf
import numpy as np
import os

from rdds.lib.tf import TextVectorizationLayer, TextPreprocessingLayer


def test_text_vectorization_layer(word_dataset, work_dir):
    """
    Test creation of dictionary and embeddings.
    """
    # GIVEN a vectorization layer, a word dataset
    # WHEN reading the dataset to assemble a dataset
    input = tf.keras.Input(shape=(1,), dtype=tf.string, name='strinput')

    text_preprocessing_layer = TextPreprocessingLayer(split_regex='splitme')
    preprocessed_dataset = word_dataset.map(map_func=text_preprocessing_layer)
    preprocessed_input = text_preprocessing_layer(input)

    text_vectorization_layer = TextVectorizationLayer()
    text_vectorization_layer.adapt(preprocessed_dataset)
    output = text_vectorization_layer(preprocessed_input)

    model = tf.keras.Model(input, output)
    model.compile()

    # THEN expect the dictionary to contain the words in the dataset
    y = model.predict(['nicely'])
    assert 'some' in text_vectorization_layer._vocabulary_layer.get_vocabulary()

    # THEN expect the embeddings to match expected shape
    assert isinstance(text_vectorization_layer.embeddings, list)
    assert isinstance(text_vectorization_layer.embeddings[0], tf.Variable)
    embeddings_matrix: np.array = text_vectorization_layer.embeddings[0].numpy()
    assert embeddings_matrix.shape == (5, 1)

    # THEN expect vocabulary to be successfully saved to file
    text_vectorization_layer.save_vocabulary_to_file('/tmp/vocab.txt')
    with open('/tmp/vocab.txt') as file:
        vocabulary: List[str] = file.read().split('\n')
        vocabulary.remove('')
        assert set(vocabulary) == set(
                ['[UNK]',
                'some',
                'nicely',
                'formatted',
                'sentence'])
    os.remove('/tmp/vocab.txt')

    tf.keras.models.clone_model(model)  # Test model clone endpoint
    tf.keras.models.Model.from_config(model.get_config())  # Test model from config endpoint

    # WHEN saving the dictionary and embeddings to file
    model.save(filepath=work_dir)
    model = None
    tf.keras.backend.clear_session()

    # THEN expect it to load successfully
    model = tf.keras.models.load_model(work_dir)
    y_loaded = model.predict(['nicely'])
    # THEN make sure it predicts the same way as the in-RAM model
    assert np.isclose((tf.reduce_sum(y - y_loaded)).numpy(), 0, atol=1E-6)
    assert len(model.get_layer('TextVectorizationVocabulary').get_vocabulary()) == 5  # UNKNW + 1
    embeddings_matrix_loaded = model.get_layer('TextVectorizationEmbeddings').weights[0].numpy()
    assert embeddings_matrix_loaded.shape == (5, 1)
    embeddings_num_diff: float = np.sum(np.subtract(embeddings_matrix, embeddings_matrix_loaded))
    assert np.isclose(embeddings_num_diff, 0, atol=1E-6)


def test_text_vectorization_mixed_size_inputs(feature_columns_dataset):
    """
    This test is about making sure text vectorization layer and post operations behave
    as expected when fed with a RaggedTensor as input.
    """
    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    # GIVEN a model using embeddings
    n_features = 2
    embedding_dimensions = 5
    input = tf.keras.Input(shape=n_features,
                           ragged=True,
                           dtype=tf.string,
                           name='input_text')
    text_preprocessing_layer = TextPreprocessingLayer()
    preprocessed_dataset = feature_columns_dataset.map(map_func=text_preprocessing_layer)
    preprocessed_input = text_preprocessing_layer(input)

    text_vectorization_layer = TextVectorizationLayer(embedding_dimensions=embedding_dimensions)
    text_vectorization_layer.adapt(preprocessed_dataset)
    # WHEN computing the embeddings for input data
    embeddings: tf.RaggedTensor = text_vectorization_layer(preprocessed_input)  # [bdim, n_features, (n_words), n_embeddings]

    model = tf.keras.Model(input, embeddings)
    model.compile()

    x = [
        ['', 'cat'],
        ['', 'champagne with cake'],
    ]

    pred: tf.RaggedTensor = model.predict(x)
    shape = pred.bounding_shape()
    for batch_idx in range(0, shape[0]):
        for feature_idx in range(0, shape[1]):
            for word_idx in range(0, shape[2]):
                for embedding_idx in range(0, shape[3]):
                    try:
                        # THEN expect that for every populated embedding dimension, i.e. value
                        # there is a real value it's properly initialized (zero centered)
                        embedding_value = pred[batch_idx, feature_idx, word_idx, embedding_idx].numpy()
                    except InvalidArgumentError:
                        # Dimension not populated, continue to next idx
                        continue
                    assert embedding_value == embedding_value
                    assert isinstance(embedding_value, np.float32), type(embedding_value)
                    # Make sure empty input embeddings are allocated to values close to zero as well
                    assert np.isclose(embedding_value, 0, atol=1E-1)


def test_save_load_vocabulary_file(vocabulary_file):
    """
    Test loading of precompiled vocabulary file.
    """
    # GIVEN a precompiled vocabulary file
    text_vectorization_layer = TextVectorizationLayer(precompiled_vocabulary_file=vocabulary_file)
    text_vectorization_layer.adapt()
    # WHEN building the layer
    # THEN expect it to load the vocabulary file
    assert text_vectorization_layer.vocabulary == ['[UNK]', 'foo', 'bar']
