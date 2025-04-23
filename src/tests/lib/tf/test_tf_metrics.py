import pytest
import tensorflow as tf
import numpy as np
from typing import Dict

"""
Test for keras metrics behavior in multi-class categorical training, eval set-up.
"""


def setup_model_check_metrics(data:  np.array,
                              labels: np.array,
                              expected_metrics: Dict[str, int]):
    """
    'Train' model and run eval step to compute metrics for a batch of (data, labels)
    :param data: Training data
    :param labels: Training labels
    :param expected_metrics: Dict of expected training-eval metrics for x, labels
    """
    # GIVEN some input data, categorical labels 2-class and some expected metrics for this data set
    assert len(expected_metrics) > 0
    # Metrics instances, must be kept locally to not leak metrics across calls
    metrics = [tf.keras.metrics.TruePositives(),
               tf.keras.metrics.TrueNegatives(),
               tf.keras.metrics.FalsePositives(),
               tf.keras.metrics.FalseNegatives(),
               tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.AUC(),
               tf.keras.metrics.Precision(),
               tf.keras.metrics.Recall()]
    # A non-trainable weight to build an E2E graph, input == output in model.
    weight = tf.Variable(initial_value=np.array([[[1, 1]]]),
                    trainable=False,
                    dtype=tf.dtypes.float32,
                    shape=tf.TensorShape((None, 1, 2)))
    input = tf.keras.Input(shape=[1, 2])
    output = tf.multiply(input, weight, name='mul')
    model = tf.keras.Model(input, output)

    def loss_fn(y_true, y_pred) -> tf.Tensor:
        return tf.keras.losses.categorical_crossentropy(y_true=y_true,
                                                        y_pred=y_pred)

    model.compile(loss=loss_fn,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=metrics)
    # WHEN training the model, and running evaluation step
    history = model.fit(x=data,
                        y=labels,
                        epochs=1,
                        steps_per_epoch=1,
                        validation_data=(data, labels))
    # THEN expect the train-eval metrics to match the expected metrics
    for metric_name_prefix in ['' 'val_']:
        # Check train as well as val metrics are identical (same data)
        for metric_name, expected_metric_value in expected_metrics.items():
            assert history.history[metric_name_prefix+metric_name][0] == expected_metric_value, \
                f'{history.history}, {metric_name} {expected_metric_value}'
    del model


# Non-expected configuration of labels, TN x 2
_CASE0 = (np.array([[[0.0, 0.0]]]),
          np.array([[[0.0, 0.0]]]),
          {'true_positives': 0,
           'true_negatives': 2,
           'false_positives': 0,
           'false_negatives': 0,
           'categorical_accuracy': 1})

_CASE1 = (np.array([[[1.0, 0.0]]]),
          np.array([[[1.0, 0.0]]]),
          {'true_positives': 1,
           'true_negatives': 1,
           'false_positives': 0,
           'false_negatives': 0,
           'categorical_accuracy': 1.0,
           'precision': 1.0,
           'recall': 1.0,
           'auc': 1.0})

_CASE2 = (np.array([[[0.0, 1.0]]]),
          np.array([[[0.0, 1.0]]]),
          {'true_positives': 1,
           'true_negatives': 1,
           'false_positives': 0,
           'false_negatives': 0,
           'categorical_accuracy': 1.0,
           'precision': 1.0,
           'recall': 1.0,
           'auc': 1.0})

# Non-expected configuration of labels, TP x 2
_CASE3 = (np.array([[[1.0, 1.0]]]),
          np.array([[[1.0, 1.0]]]),
          {'true_positives': 2,
           'true_negatives': 0,
           'false_positives': 0,
           'false_negatives': 0})

_CASE4 = (np.array([[[0.0, 1.0]]]),
          np.array([[[0.0, 0.0]]]),
          {'true_positives': 0,
           'true_negatives': 1,
           'false_positives': 1,
           'false_negatives': 0})

_CASE5 = (np.array([[[0.0, 0.0]]]),
          np.array([[[0.0, 1.0]]]),
          {'true_positives': 0,
           'true_negatives': 1,
           'false_positives': 0,
           'false_negatives': 1})


@pytest.mark.parametrize("data, labels, expected_metrics",
                         [_CASE0,
                          _CASE1,
                          _CASE2,
                          _CASE3,
                          _CASE4,
                          _CASE5])
def test_metrics(data, labels, expected_metrics):
    setup_model_check_metrics(data=data, labels=labels, expected_metrics=expected_metrics)
    tf.keras.backend.clear_session()
