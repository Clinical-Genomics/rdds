import tensorflow as tf
from typing import *
from datetime import datetime

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
        if name is None:
            name = 'TextVectorization'
        self._name = name
        self._precompiled_vocabulary_file = precompiled_vocabulary_file
        self._vocabulary_layer: tf.keras.layers.StringLookup = \
            tf.keras.layers.StringLookup(max_tokens=None,
                                         num_oov_indices=1,
                                         output_mode='int',
                                         vocabulary=precompiled_vocabulary_file,
                                         name=self._name+'Vocabulary')
        # Vocabulary config
        self._vocabulary_size: int = None
        # Embeddings config
        self._embeddings_layer: tf.keras.layers.Embedding = None

    def adapt(self,
              dataset: tf.data.Dataset = None,
              embedding_dimensions: int = 1):
        time_start: datetime = datetime.now()
        """
        "Train" vocabulary using dataset. Construct embeddings for words in dictionary.
        :param dataset: TF dataset generating vectors containing words
        :param vocabulary_file: path to a vocabulary file plain text file containing words separated by \n
        :param embedding_dimensions: The shape of embedding matrix, [N_WORDS, embedding_dimensions]
        :return:
        """
        if dataset is not None and self._precompiled_vocabulary_file is not None:
            raise ValueError('Setting both dataset and precompiled_vocabulary_file is invalid. Choose one option.')
        self._vocabulary_layer.compile()
        if dataset is not None:
            self._vocabulary_layer.adapt(dataset)
        self._vocabulary_size = len(self._vocabulary_layer.get_vocabulary(include_special_tokens=True))
        self._embeddings_layer = tf.keras.layers.Embedding(input_dim=self._vocabulary_size,
                                                           output_dim=embedding_dimensions,
                                                           name=self._name+'Embeddings')
        print(f'Creating vocabulary took {datetime.now() - time_start}')

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

    def save_vocabulary_to_file(self, file_path: str):
        vocabulary = self.vocabulary
        if len(vocabulary) == 0 or vocabulary == ['[UNK]']:
            raise ValueError(f'Tried to save empty vocabulary!')
        with open(file_path, 'w') as vocabulary_file:
            for word in vocabulary:
                print(word, file=vocabulary_file)
        print(f'Saved vocabulary to: {file_path}')

    @property
    def embeddings(self) -> List[tf.Variable]:
        """
        Return the embeddings matrix.
        :return: Matrix of floats
        """
        return self._embeddings_layer.weights
