import abc
import shap
import shap.models  # Just to get IDE checking


class ShapCompatibleSerializableModel(abc.ABC):

    """
    Helper class to specify the IO required for compatibility with SHAP lib.
    """

    @staticmethod
    @abc.abstractmethod
    def save(model: shap.models.Model, file_pointer):
        """
        :param model: A shap Model instance. The callable (model) is accessible at model.f
        :param file_pointer: An open file
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def load(file_pointer):
        """
        :param file_pointer: An open file pointer
        """
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Run model inference on input data.
        Return inferences.
        """
        pass