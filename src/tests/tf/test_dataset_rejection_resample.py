import pytest as pt
import tensorflow as tf
import numpy as np
import pytest as pt

from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator
from rdds.lib.hdf5 import Hd5DataGenerator

# Test fixtures
from tests.tf.test_data_generator import output_signature

from rdds.lib.tf import rejection_resample

@pt.fixture
def hd5_data_generator(hd5_file_path_with_categorical_labels) -> Hd5DataGenerator:
    return Hd5DataGenerator(hd5_file_path=hd5_file_path_with_categorical_labels,
                            output_tensor_format=['dataset0', 'dataset1', 'dataset2'],
                            label='label',
                            forever=False)

@pt.mark.parametrize("expected_negative_ratio, expected_positive_ratio",
                     [(1.00, 0.00),
                      (0.25, 0.75),
                      (0.50, 0.50),
                      (0.75, 0.25),
                      (0.00, 1.00)])
def test_balanced_sampling(expected_negative_ratio,
                           expected_positive_ratio,
                           hd5_data_generator,
                           output_signature):
    """
    Test for dataset to provide 50% of samples from the two classes
    """
    # GIVEN a dataset
    data_set: tf.data.Dataset = \
        get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator,
                                               output_signature=output_signature)

    data_set = data_set.cache()  # Improve testing speed by caching in RAM

    data_set = data_set.repeat(100)  # Repeat 100 times to allow somewhat accurate ratio statistics

    # WHEN sampling the dataset for a label ratio
    data_set = rejection_resample(dataset=data_set,
                                  desired_class_ratio=[expected_negative_ratio, expected_positive_ratio])
    count_positive = 0.0
    count_negative = 0.0
    for xy in data_set:
        data, labels = xy
        label, = labels
        label_negative = label[0]
        label_positive = label[1]
        if label_negative:
            count_negative += 1
        if label_positive:
            count_positive += 1
    count_samples = count_positive + count_negative
    label_negative_ratio = count_negative / count_samples
    label_positive_ratio = count_positive / count_samples
    # THEN expect the labels ratio to match the desired ratios
    assert np.isclose(label_negative_ratio, expected_negative_ratio, atol=1E-1)
    assert np.isclose(label_positive_ratio, expected_positive_ratio, atol=1E-1)
