import pandas as pd
import numpy as np
from typing import Tuple
from h5py import File as Hd5File, string_dtype, Dataset as Hd5DataSet, Group as Hd5Group

import matplotlib.pyplot as plt
from seaborn import violinplot
from rdds.lib.logging import get_logger
from ..dataset import SET_TEST, SET_TRAIN, LABEL_PATHOGENIC

LOGGER = get_logger("DatasetLoader", "info")


class DatasetLoader:
    def __init__(self,
                 path_to_hd5_dataset: str,
                 ratio_test_samples: float = 0.33,
                 seed: int = 1,
                 amount_data: float = 1.0,
                 view_data_distributions: bool = False):
        """
        :param path_to_hd5_dataset: Path to .hd5 dataset containing:
            - MIVMIR scores
            - GENMOD scores (minimal config with pedigree variant scoring)
            - Ground truth labels
        :param ratio_test_samples: Ratio of samples used for model test during training
        : param seed: Seed for generating train/test split
        :param amount_data: Reduce dataset to ratio amount (for debugging)
        :param view_data_distributions: Visualize train, test data
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

        if view_data_distributions:
            fig = plt.figure(figsize=(30, 30))
            ax = fig.add_subplot(2, 2, 1)
            train_set_benign = self._df.query(f'set == {SET_TRAIN} and pathogenic!={LABEL_PATHOGENIC}')
            train_set_pathogenic = self._df.query(f'set == {SET_TRAIN} and pathogenic=={LABEL_PATHOGENIC}')
            test_set_benign = self._df.query(f'set == {SET_TEST} and pathogenic!={LABEL_PATHOGENIC}')
            test_set_pathogenic = self._df.query(f'set == {SET_TEST} and pathogenic=={LABEL_PATHOGENIC}')
            plot_benign_train_idx = rng.permutation(train_set_benign.index.values)[0:int(np.floor(0.1 * len(train_set_benign)))]
            plot_benign_test_idx = rng.permutation(test_set_benign.index.values)[0:int(np.floor(0.1 * len(test_set_benign)))]
            # Plotting
            ax.scatter(train_set_pathogenic.score_mivmir, train_set_pathogenic.score_genmod, color='red')
            ax.scatter(test_set_pathogenic.score_mivmir, test_set_pathogenic.score_genmod, color='blue')
            ax.legend(['train', 'test'])
            ax.set_ylabel('GENMOD')
            ax.set_xlabel('MIVMIR')
            ax.title.set_text('Pathogenic samples')
            ax.set_ylim((-0.1, 1.1))
            ax.set_xlim((-0.1, 1.1))
            ax.grid(which='both', axis='both')
            ax = fig.add_subplot(2, 2, 2)
            samples_per_set_and_category = np.asarray([train_set_pathogenic.score_genmod.values,
                                                       test_set_pathogenic.score_genmod.values,
                                                       train_set_pathogenic.score_mivmir.values,
                                                       test_set_pathogenic.score_mivmir.values,
                                                       train_set_benign.score_genmod.values,
                                                       test_set_benign.score_genmod.values,
                                                       train_set_benign.score_mivmir.values,
                                                       test_set_benign.score_mivmir.values]).T
            axis_names = ['train pathogenic genmod',
                           'test patogenic genmod',
                           'train pathogenic mivmir',
                           'test pathogenic mivmir',
                           'train benign genmod',
                           'test benign genmod',
                           'train benign mivmir',
                           'test benign mivmir']
            ax.boxplot(x=samples_per_set_and_category)
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8],
                          axis_names)
            ax.title.set_text('Boxplot of samples per class, feature')
            ax = fig.add_subplot(2, 2, 3)
            ax.scatter(train_set_benign.loc[plot_benign_train_idx].score_mivmir, train_set_benign.loc[plot_benign_train_idx].score_genmod, color='red')
            ax.scatter(test_set_benign.loc[plot_benign_test_idx].score_mivmir, test_set_benign.loc[plot_benign_test_idx].score_genmod, color='blue')
            ax.legend(['train', 'test'])
            ax.set_ylabel('GENMOD')
            ax.set_xlabel('MIVMIR')
            ax.title.set_text('Benign samples')
            ax.set_ylim((-0.1, 1.1))
            ax.set_xlim((-0.1, 1.1))
            ax.grid(which='both', axis='both')
            ax = fig.add_subplot(2, 2, 4)
            violinplot(data=samples_per_set_and_category,
                       ax=ax)
            ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7],
                          axis_names)
            ax.title.set_text('Violinplot of samples per class, feature')

            fig.set_layout_engine('compressed')
            plt.show()

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
