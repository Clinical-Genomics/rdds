import tensorflow as tf
from typing import *


class TextVectorizationLayer:

    """
    Layer that implements word vocabulary and embeddings.
    """

    def __init__(self,
                 name: str = None):
        if name is None:
            name = 'TextVectorization'
        self._name = name
        self._vocabulary_layer: tf.keras.layers.StringLookup = \
            tf.keras.layers.StringLookup(max_tokens=None,
                                         num_oov_indices=1,
                                         output_mode='int',
                                         name=self._name+'Vocabulary')
        # Vocabulary config
        self._vocabulary_size: int = None
        # Embeddings config
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
                                                           name=self._name+'Embeddings')

    def __call__(self, x: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        For a tensor x, lookup words in vocabulary and return embeddings for this tensor.
        :param x: Tensor containing words
        :return: Embedding matrix for words in x
        """
        x = self._vocabulary_layer(x)
        x = self._embeddings_layer(x)
        return x

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
