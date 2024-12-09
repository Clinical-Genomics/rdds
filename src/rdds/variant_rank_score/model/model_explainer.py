import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Callable
import gc

from rdds.lib.logging import get_logger; _LOGGER = get_logger('ModelExplainer', 'debug')
from rdds.lib.model_explanation.shap import ShapExplainer
from ..dataset.class_labels import LABEL_PATHOGENIC_VARIANT
from .shap_compatible_model import ShapCompatibleModel


class ModelExplainer(ShapExplainer):

    def __init__(self,
                 model: Callable,
                 features_text: List[str],
                 features_numerical: List[str],
                 input_feature_names: List[str]):
        """
        Helper class to inferface VRS model and SHAP.

        :param model: The keras model used for inference, a callable to run inference
        :param features_text: The text feature names (as input to keras model, order important)
        :param features_numerical: The numerical feature names
        :param input_feature_names: All feature names, as input to model, for SHAP visualisation
        """
        shap_compatible_model = ShapCompatibleModel(keras_model=model,
                                                    features_numerical=features_numerical,
                                                    features_text=features_text)
        super().__init__(model=shap_compatible_model,
                         input_feature_names=input_feature_names)
        self._features_text = features_text
        self._features_numerical = features_numerical
        _LOGGER.debug(f'Initialized with features: {self._features_numerical, self._features_text}')

    def adapt(self,
              dataset: tf.data.Dataset,
              n_reference_samples: int = int(5E1)):
        _LOGGER.debug(f'Adapting to dataset')

        dataset = dataset.unbatch()  # Flatten dataset batches

        @tf.function
        def select_only_benign_samples(x, y):
            labels, = y
            # [ class_benign, class_pathogenic]
            if labels[1] == LABEL_PATHOGENIC_VARIANT:
                return False
            return True

        dataset = dataset.filter(select_only_benign_samples)
        dataset = dataset.map(lambda x, y: x)  # Drop labels
        dataset = dataset.take(n_reference_samples).as_numpy_iterator()
        data = list(dataset)  # Load to RAM
        # data : Tuple(text_features, numerical_features)
        shap_data = pd.DataFrame([text_features for text_features, _ in data],
                                 columns=self._features_text)
        shap_data = pd.concat((shap_data,
                                    pd.DataFrame([numerical_features for _, numerical_features in data],
                                                  columns=self._features_numerical)), axis=1)
        del data
        reference_data: np.ndarray = shap_data.values
        gc.collect()
        _LOGGER.debug(f'Reference data shape: {reference_data.shape}')
        super().adapt(reference_data=reference_data)
        _LOGGER.debug('Reference data adapt() complete')

    @staticmethod
    def from_saved_file(file_path: str,
                        *args,
                        **kwargs) -> ShapExplainer:
        """
        Load a ShapExplainer from file, and set a pre-loaded keras model to the model function within the SHAP framework.
        :param file_path: The file where the saved SHAP object resides
        :param args, kwargs: The arguments to ShapCompatibleModel.load_from_prior_keras_model()
        """
        return ShapExplainer.from_saved_file(file_path=file_path,
                                             model_loader=lambda file_path: ShapCompatibleModel.load_from_prior_keras_model(*args, **kwargs))
