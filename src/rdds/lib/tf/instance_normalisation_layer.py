import tensorflow as tf


@tf.keras.saving.register_keras_serializable()  # Make sure layer is available in keras save/ load operations.
class InstanceNormalisation(tf.keras.layers.BatchNormalization):

    """
    Performs instance (per feature, channel) normalisation.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'axis': [-1]})
        super().__init__(*args, **kwargs)
