from typing import *
import tensorflow as tf
import tensorflow_text as tftext


class TextPreprocessingLayer(tf.Module):

    """
    Preprocesses text input.

    Regext syntax: https://github.com/google/re2/wiki/Syntax
    """

    def __init__(self,
                 split_regex: str = '\s|\n'):
        """
        :param split_regex: Regex pattern to split text by, matches are discarded.
        """
        tf.Module.__init__(self, name='TextPreprocessing')
        self._splitter: tftext.Splitter = tftext.RegexSplitter(split_regex=split_regex)

    def _process_tensor(self, tensor):
        tensor = tf.strings.lower(tensor)  # lowercase all words
        tensor = self._splitter.split(tensor)  # split words
        return tensor

    def __call__(self, *tensors: Union[tf.RaggedTensor, Tuple[tf.RaggedTensor]]) -> Union[tf.RaggedTensor, Tuple[tf.RaggedTensor]]:
        """
        Apply text preprocessing to input tensor x
        :param tensors: RaggedTensor or Tuple[RaggedTensor]
        :return: Preprocessed data as RaggedTensor or Tuple[RaggedTensor]
        """
        if isinstance(tensors, tuple) and len(tensors) == 1:
            return self._process_tensor(tensors[0])
        processed_tensors = tuple()
        for tensor in tensors:
            processed_tensors += (self._process_tensor(tensor), )
        return processed_tensors
