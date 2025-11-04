import tensorflow as tf


@tf.keras.saving.register_keras_serializable()  # Make sure layer is available in keras save/ load operations.
class SelfAttentionLayer(tf.keras.layers.Layer):

    """
    Layer that performs self attention on input features.

    In this set up, K, Q and V are identical matrices - > self attention.

    No masking is applied, so attention is performed across all input features.
    """

    def __init__(self,
                 *args,
                 num_embedding_dimensions: int = None,
                 **kwargs):
        """
        :param args:
        :param num_embedding_dimensions: Set fixed embedding dimension or leave None for equal size to input innermost dimension
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self._num_embedding_dimensions = num_embedding_dimensions
        self._attention_layer = tf.keras.layers.Attention()

    def build(self, input_shape):
        """
        Build embeddings.
        """
        feature_dims = input_shape[-1]
        if not self._num_embedding_dimensions:
            self._num_embedding_dimensions = feature_dims
        initializer = tf.keras.initializers.GlorotNormal()
        self._embeddings = self.add_weight(name='QKV',
                                           initializer=initializer,
                                           shape=(1, self._num_embedding_dimensions, 1))

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs: Tensor of shape [batch_dim, n_features], feature dim is the attention dim
        :param args:
        :param kwargs:
        :return: Attended tensor with same shape as input and attention scores
        """
        feature_vector = inputs

        # Add middle dimension to treat outermost as batch dim in matmul, [bdim, 1, n_features]
        feature_vector = tf.expand_dims(feature_vector, -2)

        # Create Q, K, V with dimensions [batch_size, n_embeddings, n_features]
        # [1, n_embeddings, 1] x [batch_size, 1, n_features] = [batch_size, n_embeddings, n_features]
        query = tf.matmul(self._embeddings, feature_vector, transpose_a=False)
        key = tf.matmul(self._embeddings, feature_vector, transpose_a=False)
        value = tf.matmul(self._embeddings, feature_vector, transpose_a=False)
        attended_features, attention_scores = self._attention_layer([query, value, key],
                                                                    return_attention_scores=True)  #[bdim, n_embeddings, n_features]

        # Reduce the attention embeddings to create output [bdim, n_attended_features]
        # TODO: Investigate whether 2xReLu layer is helpful as well (as in original attention paper).
        # This is not a learnable reduction.
        pooling_layer = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_last')
        reduced_attended_features = pooling_layer(attended_features)

        return reduced_attended_features, attention_scores