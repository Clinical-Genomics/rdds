import tensorflow as tf
from typing import List

from . import constants


class TextVectorizationLayer:

    """
    Layer that implements word vocabulary and embeddings.
    """

    def __init__(self,
                 name: str = None):
        if not name:
            name = constants.TEXT_VECTORIZATION_LAYER_NAME
        self._name = name
        self._vocabulary_layer: tf.keras.layers.StringLookup = \
            tf.keras.layers.StringLookup(max_tokens=None,
                                         num_oov_indices=1,
                                         output_mode='int',
                                         name=f'{self._name}{constants.VOCABULARY_LAYER_NAME_SUFFIX}')
        self._vocabulary_size: int = None
        self._embeddings_layer: tf.keras.layers.Embedding = None

    def adapt(self,
              dataset: tf.data.Dataset,
              embedding_dimensions: int = 1):
        """
        "Train" vocabulary using dataset. Construct embeddings for words in dictionary.
        :param dataset: TF dataset generating vectors containing words
        :param embedding_dimensions: The shape of embedding matrix, [N_WORDS, embedding_dimensions]
        :return:
        """
        self._vocabulary_layer.compile()
        self._vocabulary_layer.adapt(dataset)
        self._vocabulary_size = len(self._vocabulary_layer.get_vocabulary(include_special_tokens=True))
        self._embeddings_layer = tf.keras.layers.Embedding(input_dim=self._vocabulary_size,
                                                           output_dim=embedding_dimensions,
                                                           name=f'{self._name}{constants.EMBEDDINGS_LAYER_NAME_SUFFIX}')

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
        :return:
        """
        return self._vocabulary_layer.get_vocabulary()

    @property
    def embeddings(self) -> List[tf.Variable]:
        """
        Return the embeddings matrix.
        :return: Matrix of floats
        """
        return self._embeddings_layer.weights
