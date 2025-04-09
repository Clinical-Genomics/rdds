import tensorflow as tf

_F1_DISCRETIZATION_THRESHOLD = 0.5


def f1(y, y_pred, threshold=_F1_DISCRETIZATION_THRESHOLD):
    """
    Compute F1 score for a single class, binary classifier.
    Label 1 considered POSITIVE and 0 as NEGATIVE.
    :param y: [1.0, 0.0, ...] dtype tf.float32
    :param y_pred: [0.25, 0.98, ...] dtype tf.float32
    :param threshold: Threshold for converting y_pred to binary label
    :return: MCC score
    """
    # Check inputs
    tf.debugging.assert_equal(tf.size(y), tf.size(y_pred))
    # Ground truth
    ground_truth_positives = tf.cast(y, tf.bool)
    ground_truth_negatives = tf.math.logical_not(ground_truth_positives)
    # Predictions
    prediction_positives = tf.math.greater_equal(y_pred, tf.constant(threshold, dtype=tf.float32))
    prediction_negatives = tf.math.logical_not(prediction_positives)
    # Confusion matrix
    tps = tf.size(tf.where(tf.math.logical_and(ground_truth_positives, prediction_positives))[:, 0])
    tns = tf.size(tf.where(tf.math.logical_and(ground_truth_negatives, prediction_negatives))[:, 0])
    fns = tf.size(tf.where(tf.math.logical_and(ground_truth_positives, prediction_negatives))[:, 0])
    fps = tf.size(tf.where(tf.math.logical_and(ground_truth_negatives, prediction_positives))[:, 0])
    # F1 score
    numerator = tf.cast(2 * tps, dtype=tf.float32)
    denominator = tf.cast((2 * tps) + fps + fns, dtype=tf.float32)
    return tf.math.divide_no_nan(numerator, denominator)
