import numpy as np
import tensorflow as tf
import pytest

from rdds.lib.tf import EmbeddingsReductionLayer, TextPreprocessingLayer

RAGGED_FULL = tf.ragged.constant(
[
    ['this is an', 'unseenword'],
    ['nice with', 'champagne and cake'],
])

RAGGED_MIXED = tf.ragged.constant(
[
    ['', 'cat'],
    ['', 'champagne and cake'],
])

RAGGED_EMPTY = tf.ragged.constant(
[
    ['', ''],
    ['', ''],
])


@pytest.mark.parametrize('x',
                         [
                             RAGGED_FULL,
                             RAGGED_MIXED,
                             RAGGED_EMPTY
                         ])
def test(feature_columns_dataset, work_dir, x):
    """
    Test for embeddings reduction layer when input is mixed length or
    not supplied. The layer should always return a tensor that is either
    filled with embeddings or zero-initialized.
    """
    tf.debugging.enable_check_numerics(True)  # Throw error on +-inf or NaN in computation

    # GIVEN a model using embeddings reduction
    n_features = 2
    embedding_dimensions = 5
    input = tf.keras.Input(shape=(n_features),  # [bdim, n_features]
                           ragged=True,
                           dtype=tf.string,
                           name='input_text')

    text_preprocessing_layer = TextPreprocessingLayer()
    preprocessed_dataset = feature_columns_dataset.map(map_func=text_preprocessing_layer)
    preprocessed_input = text_preprocessing_layer(input)  # [bdim, n_features, n_words]

    # WHEN computing the reduction for the embeddings
    embeddings_reduction_layer = EmbeddingsReductionLayer(embedding_dimensions=embedding_dimensions)
    embeddings_reduction_layer.adapt(preprocessed_dataset)
    embedding_features = embeddings_reduction_layer(preprocessed_input)

    model = tf.keras.Model(input, embedding_features)
    model.compile()

    def check_output_size_and_content(x: tf.RaggedTensor, y: tf.Tensor):
        shape = x.bounding_shape()
        # [bdim, n_features, n_words==1, embeddings]
        expected_shape = tf.TensorShape((shape[0],
                                         shape[1],
                                         1,  # Word reduction into 1 single dimension for all words
                                         embedding_dimensions))
        assert y.shape == expected_shape
        for batch in range(0, y.shape[0]):
            for feature in range(0, y.shape[1]):
                for word in range(0, y.shape[2]):
                    for embedding in range(0, y.shape[3]):
                        assert np.isclose(y[batch, feature, word, embedding], 0, atol=1E-1)

    # THEN make sure return tensor is filled with embeddings or zero initialized
    for batch in feature_columns_dataset:
        check_output_size_and_content(x=batch, y=model.predict(batch))
    check_output_size_and_content(x=x, y=model.predict(x))

    tf.debugging.disable_check_numerics()

    # THEN expect (de)-serialization of model to work well
    model.get_config()
    model.save(work_dir)
    tf.keras.models.load_model(work_dir)
