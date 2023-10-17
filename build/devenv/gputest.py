import tensorflow as tf
from tensorflow.python.platform import build_info as build

print(f"TF was built with CUDA v{build.build_info['cuda_version']}")

devices = tf.config.list_physical_devices('GPU')

assert len(devices) > 0, 'Expected to find a GPU device but found none'

