import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Callable, Tuple
import gc

from rdds.lib.logging import get_logger; _LOGGER = get_logger('ModelExplainer', 'debug')
from rdds.lib.model_explanation.shap import ShapExplainer
from ..dataset.class_labels import LABEL_PATHOGENIC_VARIANT
from .shap_compatible_model import ShapCompatibleModel


class ModelExplainer(ShapExplainer):

    def __init__(self,
                 model: Callable,
                 input_tensor_spec: Tuple[tf.TensorSpec, ...]):
        """
        Helper class to interface VRS model and SHAP.

        :param model: The keras model used for inference, a callable to run inference
        :param input_tensor_spec: The model input tensor spec (as input to keras model, order sensitive)
        """
        shap_compatible_model = ShapCompatibleModel(keras_model=model,
                                                    input_tensor_spec=input_tensor_spec)

        self._input_feature_names: List[str] = []
        for tensor_spec in input_tensor_spec:
            self._input_feature_names.append(tensor_spec.name)
        super().__init__(model=shap_compatible_model,
                         input_feature_names=self._input_feature_names)
        _LOGGER.debug(f'Initialized with features: {self._input_feature_names}')

    def adapt(self,
              dataset: tf.data.Dataset,
              n_reference_samples: int = int(5E1)):
        _LOGGER.debug(f'Adapting to dataset')

        dataset = dataset.unbatch()  # Flatten dataset batches

        @tf.function
        def drop_weights(*args):
            if len(args) == 2:
                data, label = args
            elif len(args) == 3:
                data, label, weights = args
            else:
                raise ValueError(f'Unknown input format {args}')
            return data, label

        dataset = dataset.map(drop_weights)

        @tf.function
        def select_only_benign_samples(data, label):
            pathogenicity_label, = label
            if pathogenicity_label == LABEL_PATHOGENIC_VARIANT:
                return False
            return True

        dataset = dataset.filter(select_only_benign_samples)
        dataset = dataset.map(lambda x, y: x)  # Drop labels
        dataset = dataset.take(n_reference_samples).as_numpy_iterator()
        data = list(dataset)  # Load to RAM
        # data : Tuple[Union[tf.string, tf.float32 with shape (n_reference_samples, n_features) ]])
        shap_data = pd.DataFrame(data=data, columns=self._input_feature_names)
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
