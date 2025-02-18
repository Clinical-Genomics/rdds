import numpy as np
from typing import Tuple, Callable, Dict
import tensorflow as tf
from rdds.lib.model_explanation.shap import ShapCompatibleSerializableModel


class ShapCompatibleModel(ShapCompatibleSerializableModel):

    """
    Adaptor class to Shap library.

    The model used for SHAP inference is never saved/loaded to/from file,
    this is done prior to instantiation of this class.

    The expectation is that once the SHAP is adapted to reference data using a MODEL,
    the same MODEL should be used for computing new explanations going forward.
    """

    def __init__(self,
                 keras_model: Callable,
                 input_tensor_spec: Tuple[tf.TensorSpec, ...]):
        """
        :param keras_model: A callable that accepts input data and generates inferences
        :param input_tensor_spec: Model input specification, order sensitive
        """
        self._keras_model: Callable = keras_model
        self._input_tensor_spec = input_tensor_spec

    def save(self, shap_model, file_pointer):
        # Nothing to save, rely on load_from_preloaded_keras_model()
        pass

    @staticmethod
    def load(file_pointer):
        # Nothing to load, rely on load_from_preloaded_keras_model()
        pass

    @staticmethod
    def load_from_prior_keras_model(*args, **kwargs):
        """
        Method to instantiate ShapCompatibleModel using a pre-loaded tf keras model.
        """
        return ShapCompatibleModel(*args, **kwargs)

    def _to_tensors(self, array: np.ndarray) -> Dict[str, tf.Tensor]:
        """
        Convert array of mixed-type data to separate Tensors with defined
        dtypes.
        """
        tensors: Dict[str, tf.Tensor] = dict()
        for col_idx, tensor_spec in enumerate(self._input_tensor_spec):
            tensors.update({
                tensor_spec.name: tf.constant(array[:, col_idx], dtype=tensor_spec.dtype)
            })
        return tensors

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """
        Convert a [batch_dim, n_features] matrix to Keras compatible input
        and run the model. Return inferences as np.ndarray shaped [batch_dim]
        """
        tensors = self._to_tensors(array)
        inferences: np.ndarray = self._keras_model(tensors)
        return np.array(inferences)
