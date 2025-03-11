import tensorflow as tf
import numpy as np
import pytest as pt

from rdds.lib.tf import TextAugmentDropoutDataset

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
SEED = 1


@pt.mark.parametrize('expected_dropout', np.linspace(0, 1, 10))
@pt.mark.parametrize('dropout_feature_idx', [0, 1])
def test_text_augmentation(expected_dropout, dropout_feature_idx):
    """
    Test for checking dropout augmentation in pipeline.
    """
    # GIVEN some input data sentences (2 features, 1 label)
    def data_generator():

        sentences = [
            b'foo and ?',
            b'bar',
            b'cat and lynx',
            b'cow',
            b'donkey farm',
            b'banana',
            b'coffe and cake',
            b'biscuits',
            b'grass flower apple tree',
            b'pine tree'
        ]

        for i in range(1, len(sentences)):
            yield ((sentences[i-1], sentences[i],), ((NEGATIVE_LABEL, POSITIVE_LABEL),))  # (f0, f1), (label, )

    dataset = tf.data.Dataset.from_generator(lambda: data_generator(),
                                             output_signature=(
                                                              (
                                                                tf.TensorSpec((), dtype=tf.string),
                                                                tf.TensorSpec((), dtype=tf.string),
                                                              ),
                                                              (tf.TensorSpec((2, ), dtype=tf.float32), ))
                                                              )

    # WHEN augmenting the data
    text_augment_dataset = TextAugmentDropoutDataset(target_data_tensor_idx=dropout_feature_idx,  # Only augment the Nth feature
                                                     dropout_on_categorical_label_value=POSITIVE_LABEL,
                                                     seed=SEED,
                                                     dropout_ratio=expected_dropout)
    dataset = text_augment_dataset(dataset)
    dataset = dataset.repeat(5)  # To generate accurate statistics

    n_samples = 0.0
    n_dropouts = 0.0
    for row in dataset.as_numpy_iterator():
        n_samples += 1
        features, label = row  # Unpack
        dropped_out_feature = features[dropout_feature_idx]
        # THEN expect accurate dropout of feature dropout_feature_idx
        if dropped_out_feature == text_augment_dataset._dropout_value:
            n_dropouts += 1
    actual_dropout = n_dropouts / n_samples
    assert np.isclose(actual_dropout, expected_dropout, atol=0.1), \
        f'Actual dropout != expected dropout {actual_dropout} {expected_dropout}'
