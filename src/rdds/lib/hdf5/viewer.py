from h5py import File, Group, Dataset
from typing import *
from pprint import pprint
from random import randint, seed
seed(0)


class Hdf5Viewer:

    def __init__(self,
                 file_path: str):
        self.file_path = file_path

    def __call__(self,
                 n_samples: int = 2):
        """
        Print data columns and some data samples to stdout
        :param n_samples: Samples to view
        :return:
        """

        n_data_columns: int = 0
        total_ram_consumption_MB: float = 0.0

        def print_data(name: str, object: object):
            nonlocal n_samples, h5py_file, n_data_columns, total_ram_consumption_MB

            if isinstance(object, Group):
                print(f'Group: {name}')
                return

            if isinstance(object, Dataset):
                n_data_columns += 1
                if len(object.shape) > 1:
                    raise NotImplementedError('Only 1D data supported at this time')
                sample_idxs: List[int] = [randint(0, object.shape[-1]) for _ in range(0, n_samples)]
                sample_idxs.sort()
                data_samples = object[sample_idxs]
                dataset_size_MB: float = object.nbytes / (1024.0 * 1024.0)
                total_ram_consumption_MB += dataset_size_MB
                pprint(f'{name}:{dataset_size_MB:.2f}MB {object.dtype}{object.shape}={data_samples}', width=256)

        with File(self.file_path, 'r') as h5py_file:
            h5py_file.visititems(print_data)
        print(f'Totalling {n_data_columns} data columns, estimated {total_ram_consumption_MB/1024.0:.2f}GB RAM')
