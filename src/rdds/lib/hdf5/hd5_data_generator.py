import h5py
from typing import *

from . import _LOGGER

DataSetNames = str  # A HDF5 data set name
OutputTensorFormat = Union[List[DataSetNames], List[List[DataSetNames]]]

class Hd5DataGenerator:

    """
    Class that wraps HD5 data as a Generator instance.

    * TODO: Shuffle data support
    """

    def __init__(self,
                 hd5_file_path: str,
                 group_name: str = None,
                 output_tensor_format: OutputTensorFormat = None,
                 forever: bool = True):
        """
        :param hd5_file_path: HD5 file to generate data from
        :param group_name: Group in HD5 file to read data from (reads all datasets)
        :param output_tensor_format: The format (dataset names) constituting the output tensor
        :param forever: Loop over data forever
        """
        self._hd5_file_path: str = hd5_file_path
        self._hd5_file: h5py.File = h5py.File(self._hd5_file_path, 'r')
        self._data_length = int

        if group_name is None:
            group_name = list(self._hd5_file.keys())[0]
        _LOGGER.info(f'Generating data from group \'{group_name}\'')
        self._group_name: str = group_name
        self._group: h5py.Group = self._hd5_file[self._group_name]

        if output_tensor_format is None:
            # TODO: Deterministic output format
            _LOGGER.warning('Output tensor format not deterministic and depends on HD5 content')
            output_tensor_format = list(self._group.keys())
        self._output_tensor_format: OutputTensorFormat = output_tensor_format
        _LOGGER.info(f'OutputTensorFormat: {self._output_tensor_format}')

        # Check that dataset shapes are identical.
        # Deduce first dataset name based on output_tensor_format, a possibly nested list.
        if isinstance(self._output_tensor_format[0], str):
            zeroeth_dataset = self._group[self._output_tensor_format[0]]
        elif isinstance(self._output_tensor_format[0], list):
            zeroeth_dataset = self._group[self._output_tensor_format[0][0]]
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

    def _assemble_output_vector(self,
                                output_tensor_format: OutputTensorFormat) -> Tuple[Union[str, float], ...]:
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
                output_vector += (self._group[dataset_name][self._idx], )
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

    def __call__(self) -> Tuple[Union[str, float], ...]:
        while True:
            yield self._assemble_output_vector(self._output_tensor_format)
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
