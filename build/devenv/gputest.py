import tensorflow as tf

devices = tf.config.list_physical_devices('GPU')

assert len(devices) > 0, 'Expected to find a GPU device but found none'

