import os

import pytest as pt
import tensorflow as tf
from typing import *
import tempfile
import shutil
import numpy as np

from rdds.lib.tf import TextVectorizationLayer, TextPreprocessingLayer


@pt.fixture
def work_dir():
    work_dir = tempfile.mkdtemp(dir='/tmp')
    yield work_dir
    shutil.rmtree(work_dir)


@pt.fixture
def word_dataset() -> tf.data.Dataset:
    words: List[str] = ['SomeSPLITMEnicely', 'formatted', 'sentence']
    return tf.data.Dataset.from_tensor_slices(words)


@pt.fixture
def text_preprocessing_layer():
    return TextPreprocessingLayer(split_regex='SPLITME')


def test_text_vectorization_layer(word_dataset, text_preprocessing_layer, work_dir):
    """
    Test creation of dictionary and embeddings.
    """
    # GIVEN a vectorization layer, a word dataset
    text_vectorization_layer = TextVectorizationLayer()
    # WHEN reading the dataset to assemble a dataset
    text_vectorization_layer.adapt(word_dataset.map(map_func=text_preprocessing_layer))

    input = tf.keras.Input(shape=(1,), dtype=tf.string)
    output = text_vectorization_layer(input)
    model = tf.keras.Model(input, output)
    model.compile()
    # THEN expect the dictionary to contain the words in the dataset
    y = model.predict(['nicely'])
    assert 'Some' in text_vectorization_layer._vocabulary_layer.get_vocabulary()

    # THEN expect the embeddings to match expected shape
    assert isinstance(text_vectorization_layer.embeddings, list)
    assert isinstance(text_vectorization_layer.embeddings[0], tf.Variable)
    embeddings_matrix: np.array = text_vectorization_layer.embeddings[0].numpy()
    assert embeddings_matrix.shape == (5, 1)

    # WHEN saving the dictionary and embeddings to file
    model.save(filepath=work_dir)
    model = None
    tf.keras.backend.clear_session()

    # THEN expect it to load successfully
    model = tf.keras.models.load_model(work_dir)
    y_loaded = model.predict(['nicely'])
    # THEN make sure it predicts the same way as the in-RAM model
    assert y == y_loaded
    assert len(model.get_layer('TextVectorizationVocabulary').get_vocabulary()) == 5  # UNKNW + 1
    embeddings_matrix_loaded = model.get_layer('TextVectorizationEmbeddings').weights[0].numpy()
    assert embeddings_matrix_loaded.shape == (5, 1)
    embeddings_num_diff: float = np.sum(np.subtract(embeddings_matrix, embeddings_matrix_loaded))
    assert np.isclose(embeddings_num_diff, 0, atol=1E-6)
