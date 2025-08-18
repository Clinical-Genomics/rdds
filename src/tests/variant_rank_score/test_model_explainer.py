import pytest as pt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from typing import List, Dict
import gc

from rdds.lib.determinism import SEED

from rdds.variant_rank_score.model.model_explainer import ModelExplainer
# For usage of LABEL_BENIGN_VARIANT, see generate_labels()
from rdds.variant_rank_score.dataset.class_labels import LABEL_BENIGN_VARIANT

class DummyTensorSpec:
    pass

class DummyKerasModel:

    """
    Test class to mimic TF Keras Model API.
    """

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> np.ndarray:
        """
        Return faked predictions as 1D array with size equal to batch dimension
        """
        # Deduce batch (outermost) dimension
        name = list(tensors.keys())[0]
        batch_dim = tensors[name].shape[0]
        return np.ones(batch_dim)


@pt.fixture()
def dataset() -> tf.data.Dataset:  # ((text_tensor, num_tensor), (labels, ))
    df = pd.DataFrame(data={
        't0': [f'a{i}' for i in range(0, 10)],
        't1': [f'b{i}' for i in range(0, 10)],
        'n0': np.random.random(10),
        'n1': np.random.random(10)
    })

    dataset = tf.data.Dataset.from_tensors((df.t0.values,
                                           df.t1.values,
                                           df.n0.values,
                                           df.n1.values))
    dataset = dataset.unbatch()

    @tf.function
    def generate_labels(t0, t1, n0, n1):
        return (t0, t1, n0, n1), (tf.constant([LABEL_BENIGN_VARIANT]), )

    dataset = dataset.map(generate_labels)
    dataset = dataset.batch(5)
    return dataset


def test_save_load(work_dir, dataset: tf.data.Dataset):
    """
    Test for saving and loading ModelExplainer with a pre-initialized Keras Model.
    """
    all_features = ['t0', 't1', 'n0', 'n1']

    input_tensor_spec = []
    for name in all_features:
        dummy_tensor_spec = DummyTensorSpec()
        dummy_tensor_spec.name = name
        if 't' in name:
            dummy_tensor_spec.dtype = tf.string
        elif 'n' in name:
            dummy_tensor_spec.dtype = tf.float32
        else:
            raise ValueError(name)
        input_tensor_spec.append(dummy_tensor_spec)

    # GIVEN a keras model
    dummy_model = DummyKerasModel()
    model_explainer = ModelExplainer(model=dummy_model,
                                     input_tensor_spec=input_tensor_spec)
    model_explainer.adapt(dataset=dataset,
                          n_reference_samples=10)
    file_path = os.path.join(work_dir, 'shap.model')
    # WHEN saving, restoring model_explainer from file, and to have model_explainer use pre-loaded keras model
    model_explainer.save(file_path=file_path)
    del dummy_model
    gc.collect()

    new_model = DummyKerasModel()
    model_explainer = ModelExplainer.from_saved_file(file_path=file_path,
                                                     keras_model=new_model,
                                                     input_tensor_spec=input_tensor_spec)
    # THEN expect this model to be used
    assert hex(id(model_explainer.model.f._keras_model)) == hex(id(new_model))

def test_reproducibility(dataset):
    """
    Given some inputs, see if the data is identical for one large batch
    or multiple small batches.

    Create a simple model that sums input data.
    In this way, the data feature perturbation and inference averaging is tested.
    """

    def drop_features(x, labels):
        (t0, t1, n0, n1) = x
        return (n0, n1), labels

    dataset = dataset.map(drop_features)

    input_tensor_spec = []
    for name in ['n0', 'n1']:
        dummy_tensor_spec = DummyTensorSpec()
        dummy_tensor_spec.name = name
        if 't' in name:
            dummy_tensor_spec.dtype = tf.string
        elif 'n' in name:
            dummy_tensor_spec.dtype = tf.float32
        else:
            raise ValueError(name)
        input_tensor_spec.append(dummy_tensor_spec)

    class Model:

        def __call__(self, data: Dict[str, tf.Tensor]) -> np.ndarray:
            y = data['n0'] + data['n1']
            return y

    model = Model()

    model_explainer = ModelExplainer(model=model,
                                     input_tensor_spec=input_tensor_spec)
    model_explainer.adapt(dataset=dataset,
                          n_reference_samples=10)

    results = {}
    epochs = 2
    for epoch in range(0, epochs):
        batches = []
        for batch_idx, ((n0, n1), labels) in enumerate(dataset.as_numpy_iterator()):
            X = np.concatenate((np.expand_dims(n0, 1),
                                np.expand_dims(n1, 1)), axis=1)  # X: [bdim, 2]
            explanation = model_explainer._explainer.shap_values(X=X, gc_collect=True)

            batch = {'data': X, 'explanations': explanation}
            batches.append(batch)
        results.update({epoch: batches})

    # Compare and check results
    n_batches = len(results[0])
    for batch_idx in range(0, n_batches):
        for epoch in range(0, epochs):
            ref = results[0][batch_idx]['explanations']
            trgt = results[epoch][batch_idx]['explanations']
            delta = np.subtract(trgt, ref)  # [bdim, 2]
            errors = np.abs(delta)
            is_no_error = np.isclose(errors, 0, atol=1E-2)
            assert np.all(is_no_error)
