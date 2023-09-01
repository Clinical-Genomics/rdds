import tensorflow as tf
from typing import List
from datetime import datetime
from logging import INFO

from rdds.lib.file_io import write_file
from rdds.lib.logging import get_logger
LOGGER = get_logger(name='text_vectorization')
LOGGER.setLevel(INFO)
from . import constants


# FIXME: Bug https://github.com/keras-team/keras/issues/15163 with  tensorboard

class TextVectorizationLayer:

    """
    Layer that implements word vocabulary and embeddings.
    """

    def __init__(self,
                 name: str = None,
                 precompiled_vocabulary_file: str = None):
        """
        :param name: Layer name
        :param precompiled_vocabulary_file: A plain text file path with words to be used as vocabulary, separated by '\n'
        """
        if not name:
            name = constants.TEXT_VECTORIZATION_LAYER_NAME
        self._name = name
        self._precompiled_vocabulary_file = precompiled_vocabulary_file
        self._vocabulary_layer: tf.keras.layers.StringLookup = \
            tf.keras.layers.StringLookup(max_tokens=None,
                                         num_oov_indices=1,
                                         output_mode='int',
                                         vocabulary=precompiled_vocabulary_file,
                                         name=f'{self._name}{constants.VOCABULARY_LAYER_NAME_SUFFIX}')
        self._vocabulary: List[str] = None
        self._vocabulary_size: int = None
        self._embeddings_layer: tf.keras.layers.Embedding = None

    def adapt(self,
              dataset: tf.data.Dataset = None,
              embedding_dimensions: int = 1):
        """
        "Train" vocabulary using dataset. Construct embeddings for words in dictionary.
        :param dataset: TF dataset generating vectors containing words
        :param embedding_dimensions: The shape of embedding matrix, [N_WORDS, embedding_dimensions]
        :return:
        """
        time_start: datetime = datetime.now()
        if dataset is not None and self._precompiled_vocabulary_file is not None:
            raise ValueError('Setting both dataset and precompiled_vocabulary_file is invalid. Choose one option.')
        self._vocabulary_layer.compile()
        if dataset is not None:
            self._vocabulary_layer.adapt(dataset)
        self._vocabulary = self._vocabulary_layer.get_vocabulary(include_special_tokens=True)
        self._vocabulary_size = len(self._vocabulary)
        self._embeddings_layer = tf.keras.layers.Embedding(input_dim=self._vocabulary_size,
                                                           output_dim=embedding_dimensions,
                                                           name=f'{self._name}{constants.EMBEDDINGS_LAYER_NAME_SUFFIX}')
        LOGGER.info(f'Creating vocabulary took {datetime.now() - time_start}')

    def __call__(self, ragged_tensor: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        For ragged_tensor, lookup words in vocabulary and return word embeddings for this tensor.
        :param ragged_tensor: Tensor containing words
        :return: Embedding matrix for words in ragged_tensor
        """
        vocabulary_indices: tf.RaggedTensor = self._vocabulary_layer(ragged_tensor)
        return self._embeddings_layer(vocabulary_indices)

    @property
    def vocabulary(self) -> List[str]:
        """
        Return the words in the vocabulary
        :return: The vocabulary as strings in a List
        """
        return self._vocabulary.copy()

    def save_vocabulary_to_file(self, file_path: str):
        """
        Save the vocabulary to file.
        :param file_path: File path where to store the vocabulary.
        :return:
        :raises ValueError: In case the vocabulary is empty (should not happen)
        """
        vocabulary = self.vocabulary
        if len(vocabulary) == 0 or vocabulary == ['[UNK]']:
            raise ValueError(f'Tried to save empty vocabulary!')
        write_file(file_path=file_path, contents=vocabulary)
        LOGGER.info(f'Saved vocabulary to: {file_path}')

    @property
    def embeddings(self) -> List[tf.Variable]:
        """
        Return the embeddings matrix.
        :return: Matrix of floats
        """
        return self._embeddings_layer.weights
