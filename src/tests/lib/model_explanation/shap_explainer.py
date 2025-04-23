import gc

import pytest as pt
import numpy as np
import os
import pickle

from rdds.lib.model_explanation.shap import ShapExplainer, ShapCompatibleSerializableModel


@pt.fixture()
def reference_data() -> np.ndarray:
    yield np.asarray([[0, 0],
                      [1, 0],
                      [0, 1],
                      [2, 0]])


@pt.fixture()
def test_data() -> np.ndarray:
    yield np.array([[-0.2, -5.3]])


class SimpleSerializableModel(ShapCompatibleSerializableModel):

    def __init__(self, mode: str = 'linear'):
        self._mode = mode

    def __call__(self, data: np.ndarray) -> np.ndarray:
        a = data[:, 0]
        b = data[:, 1]
        if self._mode == 'linear':
            return np.sum((a, b), axis=0)
        if self._mode == 'nonlinear':
            return np.multiply(a, b)

    @staticmethod
    def save(model, file_pointer):
        data = {'mode': model.f._mode}
        pickle.dump(data, file_pointer)

    @staticmethod
    def load(file_pointer):
        data = pickle.load(file_pointer)
        return SimpleSerializableModel(mode=data['mode'])


MODELS = [SimpleSerializableModel('linear'), SimpleSerializableModel('nonlinear')]


@pt.mark.parametrize('model', MODELS)
def test_shap_explainer(reference_data, test_data, model):
    shap_explainer = ShapExplainer(input_feature_names=['a', 'b'], model=model)

    shap_explainer.adapt(reference_data=reference_data)

    shap_values = shap_explainer.shapley_values(test_data)


def test_save_load(reference_data, test_data, work_dir):
    """
    Test for SHAP saving and restoring from file.
    """
    # GIVEN a model and ShapExplainer
    model = SimpleSerializableModel('nonlinear')
    shap_explainer = ShapExplainer(input_feature_names=['a', 'b'], model=model)
    shap_explainer.adapt(reference_data=reference_data)
    shap_values_ref = shap_explainer.shapley_values(test_data)
    file_path = os.path.join(work_dir, 'shap')
    # WHEN saving to file and restoring Explainer
    shap_explainer.save(file_path=file_path)
    del model
    del shap_explainer
    gc.collect()

    shap_explainer_loaded = ShapExplainer.from_saved_file(file_path=file_path,
                                                          model_loader=SimpleSerializableModel.load)
    loaded_shap_values = shap_explainer_loaded.shap_values(test_data)

    # THEN expect identical explanation behavior
    d = np.sum(np.abs(shap_values_ref - loaded_shap_values))
    assert np.isclose(d, 0, atol=1E-6)


