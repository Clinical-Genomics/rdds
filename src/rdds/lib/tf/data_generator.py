import tensorflow as tf
from typing import Tuple, List, Union, Iterator
from rdds.lib.hdf5 import Hd5DataGenerator


def get_tf_dataset_from_hd5_data_generator(hd5_data_generator: Hd5DataGenerator,
                                           output_signature: Tuple[Union[tf.TensorSpec, tf.RaggedTensorSpec], ...] = None,
                                           output_shapes: List[tf.TensorShape] = None) -> tf.data.Dataset:
    """
    Create a tensorflow dataset generator to read data into Tensorflow (TF) model, using
    a Hd5DataGenerator as data source.

    :param hd5_data_generator: Instance of Hd5DataGenerator
    :param output_signature: Data types, specified using tf.TensorSpec objects.
      Example: (tf.TensorSpec((), dtype=tf.string), tf.TensorSpec((), dtype=tf.float32))
    :param output_shapes: Shape of output data (optional), otherwise shapes will be
      runtime inferred from data.
    :return: Instance of tf.data.Dataset
    """
    if not isinstance(hd5_data_generator, Hd5DataGenerator):
        raise ValueError(f'Expected a Hd5DataGenerator but got {type(hd5_data_generator)}')

    def initialize_new_iterator() -> Iterator:
        """
        Helper function to create re-instantiatable datasets.

        Tensorflow re-instantiates hd5 data generator multiple times, example during initialisation, label sampling and
        when calling tf.data.Dataset.repeat().
        :return: Iterator
        """
        return hd5_data_generator()

    return tf.data.Dataset.from_generator(generator=initialize_new_iterator,
                                          output_signature=output_signature,
                                          output_shapes=output_shapes)
