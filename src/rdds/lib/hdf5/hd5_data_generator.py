import h5py
import numpy as np
from typing import Tuple, Union, List, Any

from rdds.lib.logging import get_logger
_LOGGER = get_logger(__package__)
_LOGGER.setLevel(logging.INFO)


Hdf5DataSetNames = str  # A HDF5 data set name
OutputTensorFormat = Union[List[Hdf5DataSetNames] List[List[Hdf5DataSetNames]]]


class Hd5DataGenerator:

    """
    Class that wraps HD5 data as a Generator instance.

    * TODO: Shuffle data support
    """

    def __init__(self,
                 hd5_file_path: str,
                 group_name: str = None,
                 output_tensor_format: OutputTensorFormat = None,
                 label: str = None,
                 forever: bool = True,
                 replace_nan_floats_with: float = 0.0):
        """
        :param hd5_file_path: HD5 file to generate data from
        :param group_name: Group in HD5 file to read data from (reads all datasets)
        :param output_tensor_format: The format (dataset names) constituting the output tensor
        :param label: Dataset name, if supplied, HD5DataGenerator yields (data_tensor, labels)
          where data_tensor is equivalent to output_tensor_format field.
        :param forever: Loop over data forever
        :param replace_nan_floats_with: Floating point value to replace NaN values
        :raises ValueError: In case internal dataset rank is not 1
        """
        self._hd5_file_path: str = hd5_file_path
        self._hd5_file = h5py.File(self._hd5_file_path, 'r')
        self._data_length = int
        self._replace_nan_floats_with = replace_nan_floats_with

        self._group_name = group_name if group_name else list(self._hd5_file.keys())[0]
        _LOGGER.info(f'Generating data from group \'{group_name}\'')
        self._group_name: str = group_name if group_name else list(self._hd5_file.keys())[0]
        self._group: h5py.Group = self._hd5_file[self._group_name]

        if not output_tensor_format:
            # TODO: Deterministic output format
            _LOGGER.warning('Output tensor format not deterministic and depends on HD5 content')
            output_tensor_format = list(self._group.keys())
        self._output_tensor_format: OutputTensorFormat = output_tensor_format
        self._label = label
        _LOGGER.info(f'OutputTensorFormat: {self._output_tensor_format}, Label: {self._label}')

        # Check that dataset shapes are identical.
        # Deduce first dataset name based on output_tensor_format, a possibly nested list.
        if isinstance(self._output_tensor_format[0], str):
            zeroeth_dataset: h5py.Dataset = self._group[self._output_tensor_format[0]]
        elif isinstance(self._output_tensor_format[0], list):
            zeroeth_dataset: h5py.Dataset = self._group[self._output_tensor_format[0][0]]
        else:
            raise NotImplementedError('Unknown shape of self._output_tensor_format')
        if len(zeroeth_dataset.shape) > 1:
            raise ValueError(f'Only 1D dataset shape supported at this time')
        for dataset in self._group.values():
            if dataset.shape != zeroeth_dataset.shape:
                raise ValueError(f'Not identical dataset shapes, got {dataset.shape}!={zeroeth_dataset.shape}')
        self._data_length: int = zeroeth_dataset.shape[0]
        _LOGGER.info(f'{self._data_length} samples across {len(self._group.keys())} features')

        self._idx: int = 0  # Current index for data generation

        self._forever: bool = forever

    @property
    def data_length(self):
        return self._data_length

    def _replace_nan_values(self, sample: Union[bytes, float]) -> Union[bytes, float]:
        """
        Replace NaN values with a predefined value.
        :param sample: The sample to be checked, potentially float::NaN
        :return: sample, possibly replaced.
        """
        if isinstance(sample, bytes):
            # A bytestring might be empty b'', which is OK
            return sample
        if isinstance(sample, (float, np.float64, np.float32)):
            if not sample == sample:  # NaN check
                return self._replace_nan_floats_with
            return sample
        raise NotImplementedError(f'Replacing NaN values for dtype {type(sample)} not implemented.')

    def _assemble_output_vector(self,
                                output_tensor_format: OutputTensorFormat) -> Tuple[Union[str, float], Any]:
        """
        Assemble data from hd5 file and return as tuple.
        :param output_tensor_format: A possibly nested output_tensor_format, see definition above.
        :return: Tuple of data according to output_tensor_format
        """
        output_vector: Tuple[Union[str, float]] = tuple()
        if isinstance(output_tensor_format[0], list):
            for output_tensor_format_inner in output_tensor_format:
                output_vector += (self._assemble_output_vector(output_tensor_format_inner), )
        elif isinstance(output_tensor_format[0], str):
            for dataset_name in output_tensor_format:
                sample: Any = self._group[dataset_name][self._idx]
                sample = self._replace_nan_values(sample=sample)
                output_vector += (sample, )
        return output_vector

    @property
    def data_types(self) -> Dict[str, Type]:
        """
        Return output vector data types as a dict of names to python native types.
        """
        # Map numpy dtype short names to python compatible dtypes
        numpy_dtype_lookup_map = {'d': float,  # float64
                                  'f': float,  # float32
                                  'O': bytes}  # Possibly string or other data in binary format

        def get_dtypes(output_tensor_format: List[str]) -> Dict[str, Type]:
            dtypes: Dict[str, Type] = dict()
            for dataset_name in output_tensor_format:
                dtype = numpy_dtype_lookup_map[self._group[dataset_name].dtype.char]
                dtypes.update({dataset_name: dtype})
            return dtypes

        if isinstance(self._output_tensor_format[0], list):
            dtypes: Dict[str, Type] = dict()
            for inner_list in self._output_tensor_format:
                dtypes_inner = get_dtypes(inner_list)
                for key, value in dtypes_inner.items():
                    dtypes.update({key: value})
            return dtypes
        elif isinstance(self._output_tensor_format[0], str):
            return get_dtypes(self._output_tensor_format)

    @staticmethod
    def _expand_categorical_label_1d_to_2d(label: float) -> Tuple[float, float]:
        """
        Expands a 1D label value into a multiclass 2D categorical label vector, according to the formula:
        (1 - label, label).

        This method assumes that the label is in range (0, 1).

        :param label: 1D label to be unpacked
        :return: 2D label as tuple.
        """
        if not isinstance(label, (float, np.float32)):
            raise TypeError(f'Expected a float value, got {type(label)}')
        if not 0.0 <= label <= 1.0:
            raise ValueError(f'Invalid value encountered, got {label}')
        return 1.0 - label, label

    def __call__(self) -> Tuple[Union[str, float], ...]:
        """
        Yields data from HD5 data set.
        :return:
            1. Tuple of data (tensor0, tensor1, ...) OR
            2. Tuple of tuples ((tensor0, tensor1, ...), (label, ))
        """
        while True:
            x = self._assemble_output_vector(self._output_tensor_format)
            if self._label is not None:
                label = self._assemble_output_vector([self._label])
                if len(label) > 1:
                    raise ValueError(f'Only 1D label currently supported. Got {label}')
                labels = self._expand_categorical_label_1d_to_2d(label[0])
                x = (x, ) + (labels, )  # Add label, so (data, label) is produced
            yield x
            self._idx += 1
            if self._idx >= self._data_length:
                if not self._forever:
                    _LOGGER.debug('End of epoch')
                    return
                self._idx = 0
                _LOGGER.debug('Restart epoch')

    def __del__(self):
        try:
            self._hd5_file.close()
        except AttributeError:
            pass
