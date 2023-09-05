import h5py
from typing import *

from . import _LOGGER

DataSetNames = str  # A HDF5 data set name
OutputTensorFormat = List[DataSetNames]

class Hd5DataGenerator:

    """
    Class that wraps HD5 data as a Generator instance.

    * TODO: Shuffle data support
    """

    def __init__(self,
                 hd5_file_path: str,
                 group_name: str = None,
                 output_tensor_format: OutputTensorFormat = None,
                 forever: bool = True) -> object:
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

        zeroeth_dataset = self._group[self._output_tensor_format[0]]
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

    def _assemble_output_vector(self) -> Tuple[Union[str, float], ...]:
        output_vector = tuple()
        for dataset_name in self._output_tensor_format:
            output_vector += (self._group[dataset_name][self._idx], )
        return output_vector

    def __call__(self) -> Tuple[Union[str, float], ...]:
        while True:
            yield self._assemble_output_vector()
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
