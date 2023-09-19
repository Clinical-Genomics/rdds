import tensorflow as tf
from typing import List, Tuple

from rdds.lib.determinism import SEED

_EXPECTED_N_CLASSES: int = 2

def rejection_resample(dataset: tf.data.Dataset,
                       desired_class_ratio: List[float],
                       seed: int = SEED) -> tf.data.Dataset:
    """
    Resample dataset based on class label by means of (possibly rejection or repeat) resampling.
    :param dataset: A Tensorflow Dataset instance
    :param desired_class_ratio: List of floats corresponding to the resample frequency
    :param seed: A seed for for sampling operation
    :return: A resampled dataset with desired label distribution
    """

    def class_mapping_fn(data: Tuple[tf.Tensor],
                         labels: Tuple[tf.Tensor]) -> tf.Tensor:
        """
        Deduce class from labels.
        :param data: Tensors of training data
        :param labels: Tensor of label(s), 2 class [0, 1] categorical value only supported
        :return: A boolean tensor that provides class information, 0 or 1
        """
        labels: tf.Tensor = labels[0]  # Unpack tuple
        return tf.cast(tf.math.equal(labels[0], 1), dtype=tf.int32)

    if not len(desired_class_ratio) == _EXPECTED_N_CLASSES:
        raise ValueError('Expected desired class ratio to be length 2 categorical classes')

    # FIXME: HACK for inverted class ratio returned by rejection_resample()
    desired_class_ratio[0] = 1 - desired_class_ratio[0]
    desired_class_ratio[1] = 1 - desired_class_ratio[1]

    dataset = dataset.rejection_resample(class_func=class_mapping_fn,
                                         target_dist=desired_class_ratio,
                                         seed=seed)

    # Remove additional class_mapping_fn in data
    dataset = dataset.map(map_func=lambda class_fn_value, data: data)

    return dataset