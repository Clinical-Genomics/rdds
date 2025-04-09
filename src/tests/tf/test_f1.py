import pytest as pt
import tensorflow as tf
from sklearn.metrics import f1_score as reference_f1
from numpy import isclose

from rdds.lib.tf import f1
from rdds.lib.tf.f1 import _F1_DISCRETIZATION_THRESHOLD


@pt.mark.parametrize(
    "params",
    [
        ([0.0], [0.0]),  # All TN
        ([0.0, 0.0], [0.3, 0.2]),  # All TN
        ([1.0, 1.0], [0.55, 0.8]),  # All TP
        ([1.0, 0.0], [0.7, 0.0]),  # TP, TN 1
        ([0.0, 1.0], [0.0, 1.0]),  # TP, TN 2
        ([1.0, 1.0], [0.0, 0.0]),  # FN
        ([0.0, 0.0], [1.0, 1.0]),  # FP
        ([1.0, 1.0], [1.0, 0.0]),  # TP, FN
        ([1.0, 0.0], [1.0, 0.6]),  # TP, FP
        # ... and different sizes
        ([1.0, 0.0, 1.0], [0.82, 0.6, 1.0]),
        ([1.0, 0.0, 1.0], [0.23, 0.4, 1.0]),
        ([1.0, 1.0, 0.0, 0.0], [0.23, 0.4, 1.0, 0.6]),
    ]
)
def test_f1_score(params):
    """
    Test for F1 score.
    """
    # GIVEN some data
    ground_truth, predictions = params
    discretized_predictions = predictions.copy()
    for i, v in enumerate(predictions):
        if v >= _F1_DISCRETIZATION_THRESHOLD:
            discretized_predictions[i] = 1.0
        else:
            discretized_predictions[i] = 0.0
    ground_truth = tf.constant(ground_truth, dtype=tf.float32)
    predictions = tf.constant(predictions, dtype=tf.float32)
    # WHEN computing MCC score
    # THEN make sure it's correct
    assert isclose(f1(ground_truth, predictions), reference_f1(ground_truth, discretized_predictions), atol=1E-6)


def test_f1_score_bad_size():
    """
    Test for capturing bad data shape input
    """
    import tensorflow.python.framework.errors_impl as tf_errors

    with pt.raises(tf_errors.InvalidArgumentError):
        f1([1.0, 1.0], [0.0])
