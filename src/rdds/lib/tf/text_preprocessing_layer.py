from typing import Union, Tuple
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

    def _process_tensor(self, tensor: tf.RaggedTensor) -> tf.RaggedTensor:
        tensor = tf.strings.lower(tensor)
        return self._splitter.split(tensor)

    def __call__(self, *ragged_tensors: Union[tf.RaggedTensor, Tuple[tf.RaggedTensor]]) -> Union[tf.RaggedTensor, Tuple[tf.RaggedTensor]]:
        """
        Apply text preprocessing to input tensor x
        :param ragged_tensors: RaggedTensor or Tuple[RaggedTensor]
        :return: Preprocessed data as RaggedTensor or Tuple[RaggedTensor]
        """
        if isinstance(ragged_tensors, tuple) and len(ragged_tensors) == 1:
            return self._process_tensor(ragged_tensors[0])
        processed_tensors = tuple()
        for ragged_tensor in ragged_tensors:
            processed_tensors += (self._process_tensor(ragged_tensor), )
        return processed_tensors
