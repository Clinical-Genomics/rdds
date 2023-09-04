import tensorflow as tf


class DnaSequenceTrimmer:

    """
    Removes DNA sequences from RaggedTensor bytestrings.
    """

    def __init__(self,
                 regexp: str = '^[CAGTNcagtn]+$',
                 name: str = 'DnaSeqTrimmer'):
        self._regexp = regexp
        self._name = name

    def __call__(self, tensor: tf.RaggedTensor) -> tf.RaggedTensor:
        """
        Trim (remove) DNA sequence(s) from input text RaggedTensor.
        :param tensor: The tensor to be trimmed
        :return: A tensor, same shape, where DNA sequences are replaced with b'' (i.e. removed).
          This removes the ragged tensor dimension altogether, i.e. the DNAseq is not replaced
          by empty string but the element is masked (dropped).
        """
        # Find matching substrings by regexp (boolean tensor returned)
        regexp_matches = tf.strings.regex_full_match(input=tensor,
                                                     pattern=self._regexp,
                                                     name=self._name + 'RegexpFullMatch')
        # Invert the matches to preserve non-matching strings
        mask = tf.math.logical_not(regexp_matches)
        # Mask the original tensor to drop strings matching regexp pattern
        tensor_masked = tf.ragged.boolean_mask(data=tensor,
                                               mask=mask,
                                               name=self._name + 'BooleanMask')
        return tensor_masked
