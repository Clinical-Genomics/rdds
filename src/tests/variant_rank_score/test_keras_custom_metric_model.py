import tensorflow as tf
import pytest as pt
from numpy import isclose
from rdds.lib.tf import f1, mcc


from rdds.variant_rank_score.model.keras_custom_metric_model import KerasCustomMetricModel, MetricSpec
from rdds.variant_rank_score.model.custom_metrics import RegexpF1, F1Score, MccScore

METRICS = []
METRICS.append(MetricSpec('input', F1Score))


@pt.mark.parametrize('cls', [F1Score, MccScore])
def test_score_across_batches(cls):
    """
    Test to make sure TP performance is not watered out in a TN-rich dataset.
    """
    # GIVEN a metric
    obj = cls()
    obj.update_state(y=[1], y_pred=[0.89])
    for i in range(0, 10):
        # WHEN tracking performance across batches
        obj.update_state(y=[0], y_pred=[0])
    # THEN make sure a TN rich dataset is not degrading TP performance
    assert isclose(obj.result(), 1.0, atol=1E-6)



@pt.mark.parametrize('train',
                     # batch 0
                     [((tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), tf.constant(b'USE', dtype=tf.string)),
                       tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), 1.0),
                      # batch 1
                      ((tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), tf.constant(b'SKIP', dtype=tf.string)),
                       tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), 0.0),
                      # batch 2
                      ((tf.constant([[1.0, 0.0, 1.0]], dtype=tf.float32), tf.constant(b'USE', dtype=tf.string)),
                       tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), (2.0 * 1.0) / ((2.0 * 1.0) + 1.0))
                      ])
@pt.mark.parametrize('test',
                     # batch 0
                     [((tf.constant([[0.0, 1.0]], dtype=tf.float32), tf.constant(b'USE', dtype=tf.string)),
                       tf.constant([[0.0, 1.0]], dtype=tf.float32), 1.0),
                      # batch 1
                      ((tf.constant([[0.0, 1.0]], dtype=tf.float32), tf.constant(b'USE', dtype=tf.string)),
                       tf.constant([[1.0, 1.0]], dtype=tf.float32), (2.0 * 1.0) / ((2.0 * 1.0) + 1))
                      ])
def test_across_epochs(train, test):
    """
    Test for checking metric behavior across epochs
    """
    x, y, expected_train_f1 = train
    val_x, val_y, expected_val_f1 = test
    # GIVEN a model
    input = tf.keras.Input((), dtype=tf.float32, name='input')
    input_txt = tf.keras.Input((), dtype=tf.string, name='input_string')
    boolean_matches = tf.strings.regex_full_match(input=input_txt,
                                                  pattern='(USE)')
    mask = tf.cast(boolean_matches, tf.float32)
    output = input * tf.constant(1.0) * mask
    model = tf.keras.Model(inputs=[input, input_txt],
                           outputs=[output])
    model.compile(metrics=[F1Score(), MccScore()],
                  loss=tf.keras.losses.mean_squared_error)
    train_data = tf.data.Dataset.from_tensors((x, y))
    train_data = train_data.repeat(5)
    val_data = tf.data.Dataset.from_tensors((val_x, val_y))
    val_data = val_data.repeat(5)

    # WHEN training it
    logs = model.fit(
        train_data,
        validation_data=val_data,
        verbose=0,
        epochs=2
    )
    # THEN expect metrics to be per-epoch bounded
    for epoch, f1, val_f1 in zip(logs.epoch, logs.history['F1'], logs.history['val_F1']):
        assert isclose(f1, expected_train_f1, atol=1E-3)
        assert isclose(val_f1, expected_val_f1, atol=1E-3)
