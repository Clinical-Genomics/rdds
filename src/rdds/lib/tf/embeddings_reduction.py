import tensorflow as tf

from .text_vectorization_layer import TextVectorizationLayer


class EmbeddingsReductionLayer(TextVectorizationLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, ragged_tensor: tf.RaggedTensor):
        """
        Translate input text in ragged_tensor to embeddings and reduce the word dimension.
        :param ragged_tensor: The RaggedTensor containing textual data, expects
          an input tensor with shape [bdim, feature_dim]
        :return: A reduced representation of the generated embeddings
        """

        embeddings = super().__call__(ragged_tensor)

        """
        For inner dimensions where ragged_tensor is empty, e.g. [[], ['foo bar']], pad these dimensions with zeroes.
        If this is not the case, empty embedding dimensions are filled with unallocated values from RAM.
        [[], ['foo bar']] --> [ [[0, 0], [0, 0]], [[1, 2], [3, 4]] ]

        Note that the `embeddings.to_tensor(default_value=v, ...)` padding value and the reduction technique
        are interdependent; padding with 0.0 works well with a reduce_sum operation but not
        well with reduce_prod or reduce_max.

        Additionally make sure the output shape has at least one populated word dimension
        [bdim, feature_dim, word_dim, embeddings]
        """

        shape = ragged_tensor.bounding_shape()
        word_dim = tf.maximum(shape[2], tf.constant(1, dtype=tf.int64), name='maxshape')  # Make sure at least 1 word was seen with embedding value 0.0 padded
        with tf.control_dependencies([word_dim]):
            shape_padded = tf.concat([tf.cast([shape[0]], tf.int32),  # bdim
                                      tf.cast([shape[1]], tf.int32),  # feature dim
                                      tf.cast([word_dim], tf.int32),
                                      tf.constant([self.embedding_dimension], dtype=tf.int32)],
                                      axis=-1,
                                     name='newshape')  # Embedding dims is the innermost dim
        with tf.control_dependencies([shape_padded]):
            zero_padded_embeddings: tf.Tensor = embeddings.to_tensor(default_value=tf.constant(0.0), shape=shape_padded)

        # Do the reduction
        embeddings_dim_reduced = tf.math.reduce_sum(zero_padded_embeddings, axis=2, keepdims=True)
        return embeddings_dim_reduced