from shap import Explanation, waterfall_plot
from typing import Callable, List
import numpy as np
from .shap_kernel import ShapKernel
from .shap_compatible_serializable_model import ShapCompatibleSerializableModel
from rdds.lib.logging import get_logger
_LOGGER = get_logger('ModelExplainer', 'debug')


class ShapExplainer:

    """
    Explain model inferences using Shapley values.
    """

    def __init__(self,
                 model: ShapCompatibleSerializableModel,
                 input_feature_names: List[str],
                 seed: int = 1):
        # TODO: Docstring
        self._explainer: ShapKernel = None
        self._input_feature_names = input_feature_names
        self._seed = seed
        self._model = model

    @property
    def input_feature_names(self) -> List[str]:
        return self._input_feature_names

    def adapt(self,
              reference_data: np.ndarray):
        """
        Compute references using model and reference data.
        :param reference_data: A 2-D array [samples, features] to be used as the reference data
        """

        _LOGGER.debug(f'Computing reference')
        self._explainer: ShapKernel = ShapKernel(model=self._model,
                                                 data=reference_data,
                                                 feature_names=self._input_feature_names,
                                                 seed=self._seed)
        self._explainer.masker = None  # Allow explainer.save() without errors, since explainer.masker is not set

    def shapley_values(self, sample_data: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values from data, using loaded model function and
        reference data set.
        :param sample_data: Array of input data for calling model function,
        that will be the basis for feature explanations.
        """
        return self._explainer.shap_values(X=sample_data)

    def explain_sample(self, sample_data: np.ndarray):
        """
        Explain a single sample
        :param sample_data: A 2D matric with shape [1, n_features]
        :return: An Explanation instance
        """
        rank = len(sample_data.shape)
        if rank != 2:
            raise ValueError(f'Expected a rank 2 input, got {rank}')
        batch_size = sample_data.shape[0]
        if batch_size > 1:
            raise ValueError(f'Only 1 sample at a time is supported')
        explanation: Explanation = self._explainer(sample_data)
        waterfall_plot(explanation[0, :])

    def save(self,
             file_path: str):
        """
        Save the ShapeExplainer to file.
        :param file_path: Storage path
        """
        if self._explainer is None:
            raise ValueError(f'No explainer is available for saving!')
        _LOGGER.debug(f'Saving to {file_path}')
        with open(file_path, 'wb') as file:
            self._explainer.save(out_file=file,
                                 model_saver=self._model.save,
                                 masker_saver=None)

    @staticmethod
    def from_saved_file(file_path: str,
                        model_loader: Callable):
        """
        Load a SHAP explainer from file.
        :param file_path: The file path to the serialized object
        :param model_loader: Method to use for loading/configuring callable model function
        """
        _LOGGER.debug(f'Loading from {file_path}')
        with open(file_path, 'rb') as file:
            r = ShapKernel.load(in_file=file,
                                model_loader=model_loader,
                                masker_loader=None,
                                instantiate=True)  # False here raises TypeError: '<' not supported between instances of 'type' and 'int'
            return r
