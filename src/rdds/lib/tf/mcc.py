import tensorflow as tf

_MCC_DISCRETIZATION_THRESHOLD = 0.5


def mcc(y, y_pred, threshold=_MCC_DISCRETIZATION_THRESHOLD):
    """
    Compute MCC score for a single class, binary classifier.
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
    # MCC score
    numerator = tf.cast((tps * tns) - (fps * fns), dtype=tf.float32)
    denominator = tf.cast(((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)), dtype=tf.float32) ** 0.5
    return tf.math.divide_no_nan(numerator, denominator)
