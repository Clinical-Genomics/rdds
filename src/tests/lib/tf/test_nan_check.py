import tensorflow as tf
import numpy as np
import pytest

from tensorflow.python.framework.errors_impl import InvalidArgumentError as NanAssertionError

@pytest.mark.parametrize(
    'tensor, exception',
    [
        (tf.constant(1.0), None),
        (tf.constant([[-1, 2], [3, np.nan]]), NanAssertionError),
        (tf.constant([np.inf]), NanAssertionError),
        (tf.constant(np.nan), NanAssertionError)
    ]
)
def test(tensor: tf.Tensor, exception):
    tf.debugging.enable_check_numerics()
    tf.config.run_functions_eagerly(False)  # Disable eager execution to have proper production evaluation mechanics
    """
    Test to make sure debugging behaves as expected, as it's relied upon while training.
    :param tensor: Tensor possibly containing NaN or Inf
    :param exception: The exception (possibly) raised
    """
    # WHEN enable_check_numerics() is enabled
    c = tf.constant(1.0, dtype=tensor.dtype)
    # GIVEN some tensor
    # WHEN it contains NaN/inf raise an error
    if exception is not None:
        with pytest.raises(exception):
            tensor = tensor * c
    else:
        tensor = tensor * c
    tf.debugging.disable_check_numerics()
    tf.config.run_functions_eagerly(True)
