import tensorflow as tf
from typing import List, Tuple

from rdds.lib.determinism import SEED


def rejection_resample(dataset: tf.data.Dataset,
                       desired_class_ratio: List[float],
                       seed: int = SEED) -> tf.data.Dataset:
    """

    :param dataset:
    :return:
    """

    def class_mapping_fn(data: Tuple[tf.Tensor],
                         labels: Tuple[tf.Tensor]):
        labels: tf.Tensor = labels[0]  # Unpack tuple
        print('labels', labels)
        return tf.cast(tf.math.equal(labels[0], 1), dtype=tf.int32)

    if not len(desired_class_ratio) == 2:
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