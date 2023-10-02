import tensorflow as tf


class InstanceNormalisation(tf.keras.layers.BatchNormalization):

    """
    Performs instance (per feature, channel) normalisation.
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({'axis': [-1]})
        super().__init__(*args, **kwargs)
