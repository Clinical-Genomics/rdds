import numpy as np
import tensorflow as tf
from rdds.lib.tf import SelfAttentionLayer

def test_self_attention():
    """
    Test for attention layer successful build
    """
    feature_tensor = tf.constant(np.random.random_sample((5, 20)))  # 5 batches of 20 samples
    input = tf.keras.Input((20, ), dtype=tf.float32)  # [None, 20]

    layer = SelfAttentionLayer()
    y, _ = layer(input)

    # Test forward pass
    model = tf.keras.Model(inputs=input, outputs=y)
    model.compile()
    model.summary(line_length=160)
    _ = model(feature_tensor)

def test_self_attention_batch_integrity():
    """
    Test for attention layer integrity across batch dimension
    (batches are not affected by each other).
    """
    feature_tensor = np.random.random_sample((5, 20))  # 5 batches of 20 samples
    input = tf.keras.Input((20, ), dtype=tf.float32)  # [None, 20]

    layer = SelfAttentionLayer()
    y, _ = layer(input)

    # Test forward pass
    model = tf.keras.Model(inputs=input, outputs=y)
    model.compile()
    model.summary(line_length=160)
    batch_ref = model(feature_tensor)

    # GIVEN an attention layer
    # WHEN changing input data compared to reference
    for batch_dim in range(0, 5):
        modified_batch = feature_tensor.copy()
        modified_batch[batch_dim, :] = np.random.random_sample(20)
        modified_output = model(modified_batch)
        for i in range(0, 5):
            if i == batch_dim:
                # Expected to be modified, continue
                continue
            # THEN expect output ONLY be changed in the expected changed dimension, not in any other batch
            d = batch_ref[i, :] - modified_output[i, :]
            d = np.abs(d)
            d = np.sum(d)
            assert np.isclose(d, 0, atol=1E-5)