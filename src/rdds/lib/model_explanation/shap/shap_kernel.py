import shap
from shap._serializable import Serializer, Deserializer
import pickle
from typing import Dict
import inspect


class ShapKernel(shap.KernelExplainer):

    """
    Class to wrap KernelExplainer and provide methods to serialize self.data object.
    This is not provided when loading and instantiating from serialized file (bug).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._my_data = kwargs['data']

    @staticmethod
    def _data_saver(data, file_pointer):
        d = {'data': data}
        pickle.dump(d, file_pointer)

    @staticmethod
    def _data_loader(file_pointer):
        loaded_data: Dict = pickle.load(file_pointer)
        data = loaded_data['data']
        return data

    def save(self, out_file, *args, **kwargs):
        super().save(out_file, *args, **kwargs)
        with Serializer(out_file, "ShapKernel", version=0) as s:
            # Save the data object in KernelExplainer
            s.save("data", self.data, self._data_saver)

    @classmethod
    def _instantiated_load(cls, in_file, **kwargs):
        """
        Hack method to allow providing missing data argument on KernelExplainer.__init__(data=...)
        which is otherwise missing from the loaded, from-file constructor arguments.
        The data argument is not serialized properly in the KernelExplainer class.

        Do this by loading an instance of CustomKernel instead of KernelExplainer.
        """

        obj_type = pickle.load(in_file)
        if obj_type is None:
            return None

        if not inspect.isclass(obj_type) or (not issubclass(obj_type, cls) and (obj_type is not cls)):
            raise Exception(f"Invalid object type loaded from file. {obj_type} is not a subclass of {cls}.")

        # here we call the constructor with all the arguments we have loaded
        constructor_args = obj_type.load(in_file, instantiate=False, **kwargs)

        # Load data from serialized file (after loading the parent class, order is important)
        with Deserializer(in_file, "ShapKernel", min_version=0, max_version=0) as s:
            data = s.load("data", decoder=cls._data_loader)
        constructor_args['data'] = data
        return ShapKernel(**constructor_args)

