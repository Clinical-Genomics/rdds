import pytest as pt
import tensorflow as tf
import numpy as np

from rdds.lib.tf import InstanceNormalisation


NR_FEATURES = 10
EPOCHS = 1000


def test_instance_normalisation():
    """
    Test for computing Unit Normalisation across features.
    """
    # GIVEN some data with n features
    data: np.ndarray = np.random.default_rng(seed=1).random((1, 1, NR_FEATURES))  # [batch, sample, features]
    data[0, 0, 0:2] += -5  # Add some additional bias
    data[0, 0, 2:5] += +3.5

    input_tensor = tf.keras.Input((1, NR_FEATURES), dtype=tf.float32)
    # WHEN training an instance normalisation layer
    y: tf.Tensor = InstanceNormalisation()(input_tensor)
    model = tf.keras.Model(input_tensor, y)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model.compile(loss=loss_fn,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1E-1))
    model.fit(x=data,
              y=np.ones_like(data),
              steps_per_epoch=1,
              epochs=EPOCHS,
              verbose=0)

    preds = model.predict(data)
    # THEN expect the layer to normalize per feature
    assert np.isclose(np.sum(preds), NR_FEATURES, atol=1E-1)
