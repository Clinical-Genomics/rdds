from h5py import File, Group, Dataset
from typing import List, Union
import numpy as np
from pprint import pprint
from random import randint, seed
seed(0)  # Make sure to sample data from dataset identically every time.

KIBIBYTE: float = 1024.0


class Hdf5Viewer:

    """
    Helper class to display meta information about, and iterate over datasets in a HD5 file.
    """

    def __init__(self,
                 file_path: str):
        """
        :param file_path: File path of HD5 file to view.
        """
        self.file_path = file_path

    def __call__(self,
                 n_samples: int = 2):
        """
        Print data columns and some data samples to stdout
        :param n_samples: Samples to view
        :return:
        """

        nr_data_columns: int = 0
        total_ram_consumption_MiB: float = 0.0

        def print_data(name: str, object: Union[Group, Dataset]) -> None:
            """
            Prints subset of data in Dataset, or Group name.

            Only support printing 1D data.
            :param name: The group or dataset identifier
            :param object: A Group or Dataset object
            :return:
            :raises NotImplementedError: In case data is not rank 1
            """
            nonlocal n_samples, h5py_file, nr_data_columns, total_ram_consumption_MiB

            def get_samples() -> np.ndarray:
                sample_idxs: List[int] = [randint(0, object.shape[-1]) for _ in range(0, n_samples)]
                sample_idxs.sort()
                return object[sample_idxs]

            if isinstance(object, Group):
                print(f'Group: {name}')
                return

            if isinstance(object, Dataset):
                nr_data_columns += 1
                if len(object.shape) > 1:
                    raise NotImplementedError('Only 1D data supported at this time')
                samples = get_samples()
                dataset_size_MiB: float = object.nbytes / (KIBIBYTE * KIBIBYTE)
                total_ram_consumption_MiB += dataset_size_MiB
                pprint(f'{name}:{dataset_size_MiB:.2f}MiB {object.dtype}{object.shape}={samples}', width=256)

        with File(self.file_path, 'r') as h5py_file:
            h5py_file.visititems(print_data)
        print(f'Totalling {nr_data_columns} data columns, estimated {total_ram_consumption_MiB/KIBIBYTE:.2f} GiB RAM')
