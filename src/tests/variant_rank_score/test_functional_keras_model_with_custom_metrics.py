import os.path

import pytest as pt
import tensorflow as tf
import tempfile
import shutil
from numpy import isclose

from rdds.variant_rank_score.model.functional_keras_model_with_custom_metrics import \
    FunctionalKerasModelWithCustomMetrics

from rdds.variant_rank_score.model.custom_metrics import F1Score, FrequencyFilteredF1, MetricSpec

@pt.fixture
def work_dir() -> str:
    """
    Return a temporary working directory.
    :return: A path
    """
    work_dir = tempfile.mkdtemp(dir='/tmp')
    yield work_dir
    shutil.rmtree(work_dir)


def test_model_save_load(work_dir):

    """
    Test for making sure loaded model from file behaves identical to
    trained model.
    """
    # GIVEN a model
    input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='input')
    input_frq = tf.keras.Input(shape=(1,), dtype=tf.float32, name='frq')

    # Function to approximate
    f_x = (0.45 * input) + (input_frq - input_frq) - (0.85 * input)

    output = tf.keras.layers.Dense(1)(f_x)

    metric_specs = [
        MetricSpec(
            InputTensorName='frq',
            MetricClass=FrequencyFilteredF1,
            Kwargs={'name': 'FrequencyFilteredF1'}
        )
    ]

    model = FunctionalKerasModelWithCustomMetrics(
        inputs=[input, input_frq],
        outputs=output,
        metric_specs=metric_specs
    )

    model.compile(
        metrics=[F1Score()],
        loss=tf.losses.mean_squared_error,
        optimizer=tf.optimizers.Adam(learning_rate=0.1)
    )

    # Data
    x = (
        tf.constant([[0.1], [0.5]]),
        tf.constant([[0], [0.5 / 2000]])
    )
    y = (tf.constant([[0.0], [1.0]]), )
    dataset = tf.data.Dataset.from_tensors((x, y))
    dataset = dataset.unbatch()
    # Repeat dataset and batch all data in one train step.
    # This makes metrics easy to debug (otherwise they're averaged across steps in an epoch)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(10)

    model.fit(dataset,
              epochs=25,
              verbose=0)
    # WHEN saving the model to file ...
    save_path = os.path.join(work_dir, 'model.keras')
    model.save(save_path)
    prediction = model.predict(x)

    del model
    custom_objects = {
        'FunctionalKerasModelWithCustomMetrics': FunctionalKerasModelWithCustomMetrics,
        'F1Score': F1Score
    }
    # ... and WHEN restoring it from file
    with tf.keras.saving.custom_object_scope(custom_objects):
        model = tf.keras.saving.load_model(save_path)

    # THEN expect model behavior is identical to trained model
    loaded_model_prediction = model.predict(x)
    for batch in range(0, prediction.shape[0]):
        for sample in range(0, prediction.shape[1]):
            assert isclose(loaded_model_prediction[batch, sample],
                           prediction[batch, sample],
                           atol=1E-6)