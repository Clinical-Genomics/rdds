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
    c = np.random.default_rng(seed=1).random((1, 1, NR_FEATURES))
    c[0, 0, 0:5] += -5  # Add some additional bias

    dummy_input = tf.keras.Input((1, NR_FEATURES), dtype=tf.float32)
    # WHEN training and instance normalisation layer
    y = InstanceNormalisation()(dummy_input)
    model = tf.keras.Model(dummy_input, y)

    def loss_fn(y_true, y_pred) -> tf.Tensor:
        return tf.keras.losses.mean_squared_error(y_true, y_pred)

    model.compile(loss=loss_fn,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1E-1))
    model.fit(x=c,
              y=np.ones_like(c),
              steps_per_epoch=1,
              epochs=EPOCHS,
              verbose=0)

    preds = model.predict(c)
    # THEN expect the layer to normalize per feature layer
    assert np.isclose(np.sum(preds), NR_FEATURES, atol=1E-1)
