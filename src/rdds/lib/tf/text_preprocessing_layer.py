from typing import Union, Tuple
import tensorflow as tf
import tensorflow_text as tftext


class TextPreprocessingLayer(tf.keras.layers.Layer):

    """
    Preprocesses text input.

    Regexp syntax: https://github.com/google/re2/wiki/Syntax
    """

    def __init__(self,
                 split_regex: str = '\s|\n',
                 **kwargs):
        """
        :param split_regex: Regex pattern to split text by, matches are discarded.
        """
        super().__init__(**kwargs)
        self._split_regex = split_regex
        # Due to bug https://github.com/tensorflow/text/issues/422 wrap text op in Lambda layer
        split_func = lambda tensor, split_regex_arg: tftext.RegexSplitter(split_regex=split_regex_arg).split(tensor)
        self._splitter_layer = tf.keras.layers.Lambda(function=split_func,
                                                      arguments={'split_regex_arg': self._split_regex})

    def _process_tensor(self, tensor: tf.RaggedTensor) -> tf.RaggedTensor:
        tensor = tf.strings.lower(tensor)
        return self._splitter_layer(tensor)

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
