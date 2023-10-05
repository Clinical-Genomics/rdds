import tensorflow as tf
import tensorflow_text as tftext


class TextPreprocessingLayer(tf.Module):

    """
    Preprocesses text input.

    Regexp syntax: https://github.com/google/re2/wiki/Syntax
    """

    def __init__(self,
                 split_regex: str = '\s|\n'):
        """
        :param split_regex: Regex pattern to split text by, matches are discarded.
        """
        tf.Module.__init__(self, name='TextPreprocessing')
        self._splitter: tftext.Splitter = tftext.RegexSplitter(split_regex=split_regex)

    def __call__(self, ragged_tensor: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        Apply text preprocessing to input tensor x
        :param ragged_tensor: Tensor
        :return: Preprocessed data as RaggedTensor
        """
        return self._splitter.split(ragged_tensor)
