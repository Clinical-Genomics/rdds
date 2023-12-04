import os
import pytest
import tensorflow as tf
import numpy as np
import h5py
import tempfile

from rdds.lib.tf import InstanceNormalisationLayer
from rdds.lib.hdf5 import Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator


RNG = np.random.default_rng(seed=1)
NR_FEATURES = 10
FEATURE_LOC_SCALES = [(loc, scale) for loc, scale in
                      zip(RNG.integers(low=-100, high=100, size=NR_FEATURES) / 10.0,
                          RNG.integers(low=0, high=100, size=NR_FEATURES) / 10.0)]
NR_FEATURE_SAMPLES = NR_FEATURES * 2


def generate_feature_vector(loc: float,
                            scale: float,
                            size: int) -> np.ndarray:
    """
    Helper method to generate samples from a set with mean loc and stddev scale.
    :param loc: Mean of samples
    :param scale: Stddev of samples
    :param size: Amount of samples
    :return: An array of size from a normal distribution.
    """
    return RNG.normal(loc=loc, scale=scale, size=(size, 1))


@pytest.fixture
def hd5_dataset() -> str:
    """
    Return a file path to hd5 data set file containing test data.
    - group
        - dataset 0 [NR_FEATURE_SAMPLES]
        - dataset 1 [NR_FEATURE_SAMPLES]
        - ...
    """
    feature_vectors: np.ndarray = None
    for loc, scale in FEATURE_LOC_SCALES:
        feature_vector = generate_feature_vector(loc=loc, scale=scale, size=NR_FEATURE_SAMPLES)
        if feature_vectors is None:
            feature_vectors = feature_vector
        else:
            feature_vectors = np.concatenate((feature_vectors, feature_vector), axis=1)
    temp_file = tempfile.NamedTemporaryFile(dir='/tmp', delete=False)
    hd5_file = h5py.File(name=temp_file.name, mode='w')
    group = hd5_file.create_group(name='group')
    for feature_idx in range(0, NR_FEATURES):
        feature_data = feature_vectors[:, feature_idx]
        dataset: h5py.Dataset = group.create_dataset(name=f'{feature_idx}', shape=feature_data.shape, dtype=float)
        dataset[:] = feature_data
    hd5_file.flush()
    hd5_file.close()
    yield Hd5DataGenerator(hd5_file_path=temp_file.name,
                           output_tensor_format=[[f'{feature_idx}' for feature_idx in range(0, NR_FEATURES)]])
    os.remove(temp_file.name)


@pytest.fixture
def data_set(hd5_dataset: str) -> tf.data.Dataset:
    """
    Return a tf.data.Dataset instance with HD5 data with data shape
    [batch_dim, N_FEATURES] == [1, N_FEATURES]

    :param hd5_dataset: File path to HD5 file
    :return: An instance of tf.data.Dataset containing HD5 file data
    """
    data_set = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_dataset,
                                                      output_signature=tf.TensorSpec(shape=(1, 10), dtype=tf.float32))
    data_set = data_set.batch(batch_size=1)
    yield data_set


@pytest.fixture
def data_set_nested_signature(hd5_dataset: str):
    """
    Return a tf.data.Dataset instance with HD5 data with data shape
    [batch_dim, N_FEATURES] == [1, N_FEATURES]

    :param hd5_dataset: File path to HD5 file
    :return: An instance of tf.data.Dataset containing HD5 file data
    """
    nested_signature = (tf.TensorSpec(shape=(1, 10), dtype=tf.float32), )
    data_set = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_dataset,
                                                      output_signature=nested_signature)
    data_set = data_set.batch(batch_size=1)
    yield data_set


def test_normalisation(data_set):
    """
    Test for computing normalisation across features.
    Features should be individually normalized to mean 0.0 and stddev 1.0.
    """
    # GIVEN some data with NR_FEATURES features, NR_FEATURE_SAMPLES samples per feature
    dummy_input = tf.keras.Input((1, NR_FEATURES), dtype=tf.float32)

    # WHEN training a normalisation layer that normalizes feature columns.
    layer = InstanceNormalisationLayer()
    layer.adapt_from_dataset(data=data_set)
    y = layer(dummy_input)

    def evaluate_assert_model_predicts_normalized(input: tf.keras.Input,
                                                  output: tf.Tensor):
        """
        Check normalisation layer output, assert data is normalized.
        :param input: Keras Input tensor
        :param output: normalisation layer output
        """
        def loss_fn(y_true, y_pred) -> tf.Tensor:
            # A dummy loss function never to be used
            return tf.keras.losses.mean_squared_error(y_true, y_pred)

        model = tf.keras.Model(input, output)
        model.compile(loss=loss_fn,
                      optimizer=tf.keras.optimizers.SGD(learning_rate=1E-1))

        # Re-assemble predictions
        preds: np.ndarray = np.empty((NR_FEATURE_SAMPLES, NR_FEATURES))
        for sample_idx, pred in enumerate(model.predict(data_set)):
            pred: np.ndarray = pred[0, :]  # Drop batch dimension
            for feature_idx, normalized_value in enumerate(pred):
                preds[sample_idx, feature_idx] = normalized_value

        for feature_column in range(0, NR_FEATURES):
            # THEN expect the layer results to match the expected behavior
            preds_normalized = preds[:, feature_column]
            mean: float = np.mean(preds_normalized)
            assert np.isclose(mean, 0.0, atol=1E-1)
            std: float = np.std(preds_normalized)
            assert np.isclose(std, 1.0, atol=1E-1)

    evaluate_assert_model_predicts_normalized(input=dummy_input, output=y)
    weights_file = tempfile.NamedTemporaryFile(dir='/tmp')
    layer.save_weights_to_file(weights_file.name)
    old_weights = layer.get_weights().copy()

    # Re-initialize and load weights from file to test initialisation correctness
    tf.keras.backend.clear_session()
    layer = InstanceNormalisationLayer()
    y = layer(dummy_input)  # layer must be called with input to allocate properly sized variable matrix
    layer.load_saved_weights_file(weights_file.name)
    loaded_weights = layer.get_weights().copy()
    for old_weight, loaded_weight in zip(old_weights, loaded_weights):
        if isinstance(old_weight, np.ndarray):
            for weight_idx in range(0, len(old_weight)):
                assert np.isclose(old_weight[weight_idx], loaded_weight[weight_idx], atol=1E-6)
        elif isinstance(old_weight, np.int64):
            assert np.isclose(old_weight, loaded_weight, atol=1E-6)
        else:
            raise ValueError(f'Unexpected data type {type(old_weight)}')
    evaluate_assert_model_predicts_normalized(input=dummy_input, output=y)


def test_catch_bad_signature(data_set_nested_signature):
    """
    Test for catching nested tf.data.Dataset signature.
    """
    # GIVEN an instance normalisation layer and dataset with nested signature
    layer = InstanceNormalisationLayer()
    with pytest.raises(ValueError):
        # WHEN trying to call fit
        # THEN expect this to raise ValueError
        layer.adapt_from_dataset(data_set_nested_signature)
