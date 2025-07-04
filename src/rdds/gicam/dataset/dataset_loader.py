import pandas as pd
import numpy as np
from typing import Tuple
from h5py import File as Hd5File, string_dtype, Dataset as Hd5DataSet, Group as Hd5Group

from rdds.lib.logging import get_logger
from ..dataset import SET_TEST, SET_TRAIN, LABEL_PATHOGENIC

LOGGER = get_logger("DatasetLoader", "info")


class DatasetLoader:
    def __init__(self,
                 path_to_hd5_dataset: str,
                 ratio_test_samples: float = 0.33,
                 seed: int = 1,
                 amount_data: float = 1.0):
        """
        :param path_to_hd5_dataset: Path to .hd5 dataset containing:
            - MIVMIR scores
            - GENMOD scores (minimal config with pedigree variant scoring)
            - Ground truth labels
        :param ratio_test_samples: Ratio of samples used for model test during training
        : param seed: Seed for generating train/test split
        :param amount_data: Reduce dataset to ratio amount (for debugging)
        """
        self._path_to_hd5_dataset = path_to_hd5_dataset
        self._ratio_test_samples = ratio_test_samples
        self._seed = seed
        self._amount_data = amount_data
        hd5_file = Hd5File(self._path_to_hd5_dataset, 'r')
        mivmir_scores = hd5_file['gicamdata/mivmir'][()]
        genmod_scores = hd5_file['gicamdata/genmod'][()]
        labels = hd5_file['gicamdata/causative'][()]
        hd5_file.close()
        if amount_data < 1:
            random_idx = np.random.permutation(np.arange(len(mivmir_scores)))[0:int(len(mivmir_scores) * amount_data)]
            mivmir_scores = mivmir_scores[random_idx]
            genmod_scores = genmod_scores[random_idx]
            labels = labels[random_idx]
        assert len(mivmir_scores) == len(genmod_scores) == len(labels)
        self._df = pd.DataFrame({'score_mivmir': mivmir_scores,
                                 'score_genmod': genmod_scores,
                                 'pathogenic': labels})
        LOGGER.info(f"Loaded {len(self._df)} samples from {self._path_to_hd5_dataset}")
        # Divide into train and test set
        rng = np.random.default_rng(seed=self._seed)
        n_test_samples = int(np.floor(self._ratio_test_samples * len(self._df)))
        idx = np.arange(len(self._df))
        test_idx = rng.permutation(idx)[0:n_test_samples]
        self._df['set'] = SET_TRAIN
        self._df['set'][test_idx] = SET_TEST
        self._dlen_train = len(self._df[self._df.set == SET_TRAIN])
        self._dlen_test = len(self._df[self._df.set == SET_TEST])

    def __str__(self):
        s = ''
        s += f"input_file: {self._path_to_hd5_dataset}\n"
        s += f"train/test ratio: {self._ratio_test_samples}\n"
        s += f"seed: {self._seed}\n"
        s += f"amount data: {self._amount_data}\n"
        s += f"Dlen train: {self._dlen_train}\n"
        s += f"Dlen test: {self._dlen_test}"
        return s

    @property
    def dlen_train(self):
        return self._dlen_train

    @property
    def dlen_test(self):
        return self._dlen_test

    @property
    def amount_train_pathogenic_samples(self):
        df = self._df[self._df.set == SET_TRAIN]
        return len(df[df.pathogenic == LABEL_PATHOGENIC])

    def _get_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]], set_type: int
    ):
        data_spec, label_spec = input_spec
        df = self._df[self._df.set == set_type]
        x = df[list(data_spec)].values
        y = df[list(label_spec)].values
        return x, y

    def get_train_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]]
    ) -> Tuple[np.ndarray]:
        return self._get_data(input_spec=input_spec, set_type=SET_TRAIN)

    def get_test_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]]
    ) -> Tuple[np.ndarray]:
        return self._get_data(input_spec=input_spec, set_type=SET_TEST)

    def export_to_hd5(self, file_path: str):
        """
        Export training data to file
        """
        hd5_file = Hd5File(file_path, 'w')
        hd5_file.create_group('gicamdata')
        hd5_file.create_dataset('gicamdata/mivmir', dtype=np.float32, data=self._df.score_mivmir.values)
        hd5_file.create_dataset('gicamdata/genmod', dtype=np.float32, data=self._df.score_genmod.values)
        hd5_file.create_dataset('gicamdata/causative', dtype=np.float32, data=self._df.pathogenic.values)
        hd5_file.flush()
        hd5_file.close()
        print(f"Wrote {file_path}")
