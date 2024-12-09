import pytest as pt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from typing import List
import gc

from rdds.variant_rank_score.model.model_explainer import ModelExplainer
# For usage of LABEL_BENIGN_VARIANT, see generate_labels()
from rdds.variant_rank_score.dataset.class_labels import LABEL_BENIGN_VARIANT


class DummyKerasModel:

    """
    Test class to mimic TF Keras Model API.
    """

    def __call__(self, tensors: List[tf.Tensor]) -> np.ndarray:
        """
        Return faked predictions as 1D array with size equal to batch dimension
        """
        batch_dim = tensors[0].shape[0]
        return np.ones(batch_dim)


@pt.fixture()
def dataset() -> tf.data.Dataset:  # ((text_tensor, num_tensor), (labels, ))
    df = pd.DataFrame(data={
        't0': [f'a{i}' for i in range(0, 10)],
        't1': [f'b{i}' for i in range(0, 10)],
        'n0': np.random.random(10),
        'n1': np.random.random(10)
    })

    dataset = tf.data.Dataset.from_tensors((df[['t0', 't1']].values, df[['n0', 'n1']].values))
    dataset = dataset.unbatch()

    @tf.function
    def generate_labels(t0, t1):
        # [benign_class, pathogenic_class], below is considered LABEL_BENIGN_VARIANT
        return (t0, t1), (tf.constant([1.0, 0.0]), )

    dataset = dataset.map(generate_labels)
    dataset = dataset.batch(5)
    return dataset


def test_save_load(work_dir, dataset: tf.data.Dataset):
    """
    Test for saving and loading ModelExplainer with a pre-initialized Keras Model.
    """
    features_text = ['t0', 't1']
    features_numerical = ['n0', 'n1']
    all_features = ['t0', 't1', 'n0', 'n1']
    # GIVEN a keras model
    dummy_model = DummyKerasModel()
    model_explainer = ModelExplainer(model=dummy_model,
                                     features_text=features_text,
                                     features_numerical=features_numerical,
                                     input_feature_names=all_features)
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
                                                     features_text=features_text,
                                                     features_numerical=features_numerical)
    # THEN expect this model to be used
    assert hex(id(model_explainer.model.f._keras_model)) == hex(id(new_model))
