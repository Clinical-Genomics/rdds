import numpy as np
from typing import List, Tuple, Callable
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
                 features_text: List[str],
                 features_numerical: List[str]):
        self._keras_model: Callable = keras_model
        self._features_text = features_text
        self._features_numerical = features_numerical

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

    def _to_tensors(self, array: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Convert array of mixed-type data to separate Tensors with defined
        dtypes.
        """
        txt_batch = array[:, 0:len(self._features_text)]
        numerical_batch = array[:, len(self._features_text):]
        txt_tensor = tf.constant(txt_batch, dtype=tf.string)
        num_tensor = tf.constant(numerical_batch, dtype=tf.float32)
        return txt_tensor, num_tensor

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """
        Convert a [batch_dim, n_features] matrix to Keras compatible input
        and run the model. Return inferences as np.ndarray shaped [batch_dim]
        """
        tensors = self._to_tensors(array)
        inferences: np.ndarray = self._keras_model(*tensors)
        return np.array(inferences)
