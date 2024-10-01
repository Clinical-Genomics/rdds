import h5py
import numpy as np
from typing import List


def split_to_train_test_sets(hd5_file_path: str,
                             group_name: str,
                             ratio_test: float):
    """
    Split group_name into two new groups /train and /test transferring
    all datasets.

    Data samples are randomly selected from the datasets.

    :param hd5_file_path: The HD5 file to work with
    :param group_name: The group to be split (all datasets are transferred)
    :param ratio_test: The ratio of the test dataset [0, 1].
    :return:
    """
    hd5_out_file = h5py.File(hd5_file_path, 'r+')
    group: h5py.Group = hd5_out_file[group_name]

    # Setup array with sample index that will be split into train, test
    dlen = group[list(group.keys())[0]].shape[0]
    sample_idx = np.arange(0, dlen)
    rng: np.random.Generator = np.random.default_rng(seed=0)
    rng.shuffle(sample_idx)
    split_idx: int = int(np.ceil(ratio_test * dlen))
    sample_idxs_test: np.ndarray = np.sort(sample_idx[0:split_idx])
    sample_idxs_train: np.ndarray = np.sort(sample_idx[split_idx:])

    datasets_to_split: List[str] = list(group.keys())
    for group_name, dataset_idx in zip(['train', 'test'], [sample_idxs_train, sample_idxs_test]):
        group_split = hd5_out_file.create_group(name=group_name)
        for dataset_name in datasets_to_split:
            group_split.create_dataset(name=dataset_name,
                                       shape=(len(dataset_idx)),
                                       dtype=group[dataset_name].dtype)
            data = group[dataset_name][:]  # Do slicing on array in RAM for performance
            group_split[dataset_name][:] = data[dataset_idx]
    hd5_out_file.flush()
    hd5_out_file.close()
