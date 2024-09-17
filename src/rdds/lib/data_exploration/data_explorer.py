import h5py
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import os

from rdds.lib.workdir import get_workdir_path
from rdds.lib.logging import get_logger
from rdds.lib.list_dir import list_dir

sb.set_theme(font_scale=0.5)

_WORKDIR = get_workdir_path('data_exploration')
_LOGGER = get_logger('data_exploration', 'info')

IGNORE_FEATURES = [
    'ID',
    'POS',
    'REF',
    'ALT',
    'Annotation',
    'CSQ_Allele',
    ]

# FIXME: Call data analysis from rank variant model instead, and supply these values
FEATURES_TEXT = ['CSQ_PolyPhen',
                 'CSQ_SIFT',
                 'CSQ_CLINVAR_CLNREVSTAT',
                 'CSQ_CLINVAR_CLNSIG',
                 #'FILTER',
                 #'most_severe_consequence', # FIXME: Contains lots of DNA nukleotide data, that must be removed prior analysis
                 #'GeneticModels_model'  # Dropped since clinvar does not contain pedigree information
                 ]

FEATURES_FLOAT = ['CSQ_MaxEntScan_alt',
                  'CSQ_MaxEntScan_diff',
                  'CSQ_MES-SWA_acceptor_alt',
                  'CSQ_MES-SWA_donor_alt',
                  'CSQ_MES-SWA_donor_diff',
                  'CSQ_SpliceAI_pred_DS_AL',
                  'CSQ_SpliceAI_pred_DS_DG',
                  'CSQ_SpliceAI_pred_DS_DL',
                  'CSQ_REVEL_score',
                  'CSQ_LoFtool',
                  'CSQ_GERP++_RS',
                  'CSQ_phastCons100way_vertebrate',
                  'CSQ_phyloP100way_vertebrate',
                  'CADD',
                  'ModelScore_value',
                  'SWEGENAF',
                  'GNOMADAF_popmax',
                  'SPIDEX',
                  'CSQ_SpliceAI_pred_DS_AG',
                  #'MTAF',
                  'Frq'
                  ]
DEFAULT_FEATURE_NAMES = FEATURES_TEXT
DEFAULT_FEATURE_NAMES.extend(FEATURES_FLOAT)


class DataExplorationException(Exception): pass
class DataNotFoundException(DataExplorationException): pass
class NonNumericalDataException(DataExplorationException): pass
class NonTextualDataException(DataExplorationException): pass


class DataExplorer:

    """
    Visualize dataset by means of graphical tools.
    """

    def __init__(self,
                 hd5_file_path: str,
                 feature_names: List[str] = DEFAULT_FEATURE_NAMES,  # FIXME
                 feature_names_ignore: List[str] = ['ID', 'POS', 'REF', 'ALT', 'Annotation'],
                 data_set_names: List[str] = ['train', 'test'],
                 ground_truth_feature_name: str = 'label',
                 view_plot_immediately: bool = False,
                 fig_size: Tuple[int, int] = (25, 10)):
        """
        NOTE: Upon initialisation, this class will clear _WORKSPACE from images.

        :param hd5_file_path: File path to HDF5 file to analyse
        :param data_set_names: List of hdf5 groups to analyze
        """
        self._hd5_file_path = hd5_file_path
        self._hd5_file = h5py.File(self._hd5_file_path, 'r')
        self._feature_names = feature_names
        self._feature_names_ignore = feature_names_ignore
        self._data_set_names = data_set_names
        self._ground_truth_feature_name = ground_truth_feature_name
        self._view_plot_immediately = view_plot_immediately
        self._fig_size = fig_size

        try:
            os.mkdir(_WORKDIR)
        except (FileExistsError):
            files = list(list_dir(directory_path=_WORKDIR))
            [os.remove(file_name) for file_name in files]
            _LOGGER.info(f'Removed stale files in {_WORKDIR}')
        _LOGGER.info(f'Storing images in {_WORKDIR}')

        if not self._feature_names:
            # Analyze all features in dataset
            self._feature_names = list(self._get_feature_names())
        _LOGGER.info(f'Analyzing features: {self._feature_names}')

        _LOGGER.info(f'Ignoring features: {self._feature_names_ignore}')
        for name in self._feature_names_ignore:
            try:
                self._feature_names.remove(name)
            except ValueError:
                pass

    def _get_feature_names(self) -> Set[str]:
        """
        Get list of all feature names in data sets
        :returns: Set of feature names across data sets
        """
        feature_names: Set[str] = set()
        for data_set_name in self._data_set_names:
            feature_names.update(set(list(self._hd5_file[data_set_name].keys())))
        return sorted(feature_names)

    def _get_data(self,
                  feature_name: str,
                  group_name: str) -> np.ndarray:
        """
        Return feature data from group.
        :param feature_name: Name of feature to fetch data for
        :param group_name: HDF5 group name
        :returns: A np array with data
        :raises KeyError in case data is missing:
        """
        try:
            data: np.ndarray = self._hd5_file[group_name][feature_name][:]
            data_size: int = np.size(data)
            _LOGGER.debug(f'{group_name}/{feature_name}:{data_size} samples')
            if data_size == 0:
                raise DataNotFoundException(f'No data for {group_name}/{feature_name}')
        except KeyError as e:
            _LOGGER.fatal(f'Available features: {self._get_feature_names()}')
            raise e
        return data


    @staticmethod
    def _merge_dataset_labels(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Merge labels, data_set columns into dataset_labels column.
        :param data_frame: Dataframe containing data, label, data_set columns
        """
        data_frame['dataset_labels'] = data_frame.data_set + '-' + data_frame.labels.astype(str)
        column_names = list(data_frame.columns)
        column_names.remove('data_set')
        column_names.remove('labels')
        return data_frame[column_names]


    def _get_numerical_data(self, merge_dataset_labels: bool = False) -> pd.DataFrame:
        """
        Return all numerical data for selected features.
        :param merge_dataset_labels: Merge dataset, label columns to new column
        :returns: pd.DataFrame of numerical features
        """
        complete_data_frame: pd.DataFrame = None
        for data_set_name in self._data_set_names:
            data_set_data_frame: pd.DataFrame = None
            labels = self._get_labels(group_name = data_set_name)
            for feature_name in self._feature_names:
                try:

                    feature_data = self._get_data(feature_name=feature_name,
                                                  group_name=data_set_name)
                    data_set_size: float = float(int(len(feature_data)))
                    self._is_numerical_data(feature_data=feature_data)
                except (NonNumericalDataException) as e:
                    _LOGGER.debug(f'Skipping {data_set_name}/{feature_name}: {e}')
                    continue
                data_frame = pd.DataFrame(data={feature_name: feature_data})
                data_set_data_frame = pd.concat((data_set_data_frame, data_frame), axis=1) if data_set_data_frame is not None else data_frame
            data_set_data_frame['labels'] = labels
            data_set_data_frame['data_set'] = data_set_name
            complete_data_frame = pd.concat((complete_data_frame, data_set_data_frame), axis=0) if complete_data_frame is not None else data_set_data_frame
        if merge_dataset_labels:
            return self._merge_dataset_labels(data_frame=complete_data_frame)
        return complete_data_frame

    def _get_labels(self,
                    group_name: str) -> np.ndarray:
        """
        Return ground truth labels from group.
        :param group_name: HDF5 group name
        :returns: A np array with data
        """
        return self._get_data(feature_name = self._ground_truth_feature_name,
                              group_name = group_name)


    @staticmethod
    def _is_numerical_data(feature_data: np.ndarray):
        """
        Check data is numerical type.
        :param feature_data: Data to check
        :raises NonNumericalDataException: In case data is not numerical type
        """
        if not isinstance(feature_data[0], (float, int, np.float32, np.float64)):
            raise NonNumericalDataException(f'Contained non-numerical data: {feature_data}')


    @staticmethod
    def _is_textual_data(feature_data: np.ndarray):
        """
        Check data is text type.
        :param feature_data: Data to check
        :raises NonTextualDataException: In case data is not text type
        """
        if not isinstance(feature_data[0], (bytes)):
            raise NonTextualDataException(f'Contained non-textual data: {feature_data}')


    def boxplot(self):
        for feature_name in self._feature_names:
            try:
                complete_data_frame: pd.DataFrame = None
                for data_set_name in self._data_set_names:
                    _LOGGER.debug(f'Boxplotting feature {data_set_name}/{feature_name}')
                    labels = self._get_labels(group_name = data_set_name)  # FIXME: Optimize labels getter
                    feature_data = self._get_data(feature_name=feature_name,
                                                  group_name=data_set_name)
                    self._is_numerical_data(feature_data=feature_data)
                    data_frame = pd.DataFrame(data={feature_name: feature_data,
                                                    'labels': labels,
                                                    'dataset': [data_set_name] * len(feature_data)})
                    complete_data_frame = pd.concat([complete_data_frame, data_frame], axis=0)
                if complete_data_frame is None:
                    raise DataNotFoundException(f'Found no numerical data for feature {feature_name}')
                complete_data_frame['dataset_labels'] = complete_data_frame.dataset + '-' + complete_data_frame.labels.astype(str)
                fig = plt.figure(figsize=self._fig_size)
                ax0 = fig.add_subplot(2, 1, 1)
                sb.boxplot(y=complete_data_frame[feature_name].values,
                           x=complete_data_frame.labels.values,
                           hue=complete_data_frame.dataset.values,
                           ax=ax0)
                ax1 = fig.add_subplot(2, 1, 2)
                ax1.legend(loc='upper right')
                sb.histplot(x=complete_data_frame[feature_name].values,
                            hue=complete_data_frame.dataset_labels.values,
                            bins=1000,
                            ax=ax1)
                plt.suptitle(feature_name)
                fname: str = f'{os.path.join(_WORKDIR, feature_name)}-boxplot.png'
                plt.savefig(fname=fname)
                if self._view_plot_immediately:
                    plt.show()
            except (NonNumericalDataException, DataNotFoundException) as e:
                _LOGGER.debug(f'No boxplot for {feature_name}: {e}')
                continue

    def scatterplot(self, n_samples: int = int(1E5), max_features_per_plot: int = 5):
        """
        Plot side-by-side scatter plot of all numerical features.
        :param n_samples: Amount of samples to use for visualisation, randomized subset
        :param max_features_per_plot: Amount of features to visualize per matplotlib Figure.
        """
        complete_data_frame = self._get_numerical_data(merge_dataset_labels=True)
        if n_samples:
            np.random.seed(0)
            index_subset = np.random.randint(low=0, high=len(complete_data_frame), size=n_samples)
            complete_data_frame = complete_data_frame.iloc[index_subset]

        complete_data_frame.index = np.arange(0, len(complete_data_frame))

        if max_features_per_plot:
            column_names = list(complete_data_frame.columns)
            for split_idx in range(max_features_per_plot, len(column_names), max_features_per_plot):
                column_names_subset = list(set(column_names[split_idx - max_features_per_plot:split_idx] + ['dataset_labels']))
                sb.pairplot(data=complete_data_frame[column_names_subset], markers='.', hue='dataset_labels',kind='scatter')
                fname: str = os.path.join(_WORKDIR, f'scatter-{split_idx}.png')
                plt.savefig(fname=fname)
        else:
            sb.pairplot(data=complete_data_frame, markers='.', hue='dataset_labels',kind='scatter')
            fname: str = os.path.join(_WORKDIR, 'scatter.png')
            plt.savefig(fname=fname)
        if self._view_plot_immediately:
            plt.show()



    @staticmethod
    def _count_word_occurrence(word: bytes,
                               data: np.ndarray,
                               labels: np.ndarray) -> pd.DataFrame:
        """
        Count occurence of word b'foo' in large set of array [b'foo', b'bar', b'foo-bar', ...]
        on a per-label basis.
        Word might be represented as part of a sentence, which counts as an occurence.
        :param word: The word to count occurences for
        :param data: np.ndarray of data, 1D, byte content
        :param labels: np.ndarray of labels, 1D, byte content
        :param data_set: The data set name used for computing word count
        :raises ValueError: in case of shape mismatch data-label
        :returns: Dataframe of [word, count in label, label]
        """
        if data.shape != labels.shape:
            raise ValueError(f'Expected data, labels to have same shape: {data.shape}!={labels.shape}')
        word: str = word.decode('utf-8')
        unique_labels: Set[int] = set(labels)
        counts: Dict = dict()
        [counts.update({label: 0}) for label in unique_labels]
        for row_bytes, label in zip(list(data), list(labels)):
            row_str: str = row_bytes.decode('utf-8')
            if word in row_str:
                counts[label] += 1
        complete_data_frame: pd.DataFrame = None
        for unique_label in unique_labels:
            data_frame_label = pd.DataFrame(data={'word_sentence': [word],
                                                  'occurrence': [counts[unique_label]],
                                                  'labels': [unique_label]
                                                  })
            if complete_data_frame is None:
                complete_data_frame = data_frame_label
            else:
                complete_data_frame = pd.concat((complete_data_frame, data_frame_label), axis=0)
        return complete_data_frame


    def vocabulary_occurrence(self,
                              as_percentage_of_dataset_size: bool = False):
        """
        Count words occurring in dataset and visualize by box plots.

        # TODO: Truncate long sentences?
        :param as_percentage_of_dataset_size: Divide word count by dataset size,
          helpful for comparing different-sized dataset contents.
        """
        for feature_name in self._feature_names:
            try:
                complete_data_frame: pd.DataFrame = None
                for data_set_name in self._data_set_names:
                    _LOGGER.debug(f'Counting word occurence for feature {data_set_name}/{feature_name}')
                    labels = self._get_labels(group_name = data_set_name)  # FIXME: Optimize labels getter
                    feature_data = self._get_data(feature_name=feature_name,
                                                  group_name=data_set_name)
                    data_set_size: float = float(int(len(feature_data)))
                    self._is_textual_data(feature_data=feature_data)
                    unique_words_or_sentences: Set[byte] = set(feature_data)  # Might contain single, multiple words
                    unique_labels: Set[byte] = set(labels)
                    for word_sentence in unique_words_or_sentences:
                        df_counts: pd.DataFrame = self._count_word_occurrence(word=word_sentence, data=feature_data, labels=labels)
                        df_counts['dataset'] = data_set_name
                        if as_percentage_of_dataset_size:
                            df_counts['occurrence'] = 100.0 * (df_counts['occurrence'].values / data_set_size)
                        if complete_data_frame is None:
                            complete_data_frame = df_counts.copy()
                        else:
                            complete_data_frame = pd.concat((complete_data_frame, df_counts), axis=0)
            except (NonTextualDataException) as e:
                _LOGGER.debug(f'No word occurence for {feature_name}: {e}')
                continue

            # Assemble a mixed column of dataset-label to allow grouping
            complete_data_frame['dataset_labels'] = complete_data_frame.dataset + '-' + complete_data_frame.labels.astype(str)
            fig = plt.figure(figsize=self._fig_size)
            ax = fig.add_subplot()
            sb.barplot(x=complete_data_frame.word_sentence,
                         y=complete_data_frame.occurrence,
                         hue=complete_data_frame.dataset_labels,
                         errorbar=None,
                         ax=ax)
            suptitle: str = f'{feature_name} word count'
            suptitle += '\nas % of dataset)' if as_percentage_of_dataset_size else ''
            plt.suptitle(suptitle)
            fname: str = f'{os.path.join(_WORKDIR, feature_name)}-word-occurrence.png'
            fname = fname.replace('.png', '-normalized.png') if as_percentage_of_dataset_size else fname
            plt.savefig(fname=fname)
            if self._view_plot_immediately:
                plt.show()

    def feature_correlation(self):
        """
        Visualise feature correlation by means of correlation matrix.
        """
        data_frame = self._get_numerical_data(merge_dataset_labels=True)
        feature_columns = list(data_frame.columns)
        feature_columns.remove('dataset_labels')
        data_frame_corr = data_frame[feature_columns].corr()
        mask = np.triu(np.ones_like(data_frame_corr, dtype=bool))
        cmap = sb.diverging_palette(230, 20, as_cmap=True)
        fig = plt.figure(figsize=self._fig_size)
        ax = fig.add_subplot()
        sb.heatmap(data_frame_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.suptitle('Feature correlation')
        fname: str = os.path.join(_WORKDIR, 'correlation.png')
        plt.savefig(fname=fname)
        if self._view_plot_immediately:
            plt.show()

    def count_label_ratio(self):
        """
        Compute the ratio of labels in every data set.
        """
        print('## Count label ratio ##')
        data_frame = self._get_numerical_data()
        for data_set in data_frame['data_set'].unique():
            labels = data_frame[data_frame.data_set == data_set]['labels']
            if labels.hasnans:
                raise ValueError(f'Data set {data_set} labels contains NaNs! This should not happen.')
            labels_unique_values = labels.unique()
            print(data_set, labels.value_counts(dropna=False))
            print(data_set, labels.value_counts(dropna=False, normalize=True))

    def print_data_to_stdout(self):
        """
        Print numerical raw data to stdout for visual inspection.
        """
        data = self._get_numerical_data()
        linewidth = 300
        old_printopts = np.get_printoptions()
        np.set_printoptions(precision=6, linewidth=linewidth)
        for i, row in enumerate(data.itertuples()):
            data_str = f'{np.array(row)}'
            data_str_len = len(data_str)
            remainder = linewidth - data_str_len
            if row.labels == 1.0:
                data_str += '#' * remainder
            print(data_str)
            if i % 100 == 0:
                print(data.columns.values)
        np.set_printoptions(old_printopts)

    def __call__(self):
        """
        Visualize dataset.
        """

        self.boxplot()
        self.scatterplot()
        self.vocabulary_occurrence()
        self.vocabulary_occurrence(as_percentage_of_dataset_size=True)
        self.feature_correlation()
        self.count_label_ratio()

    def __del__(self):
        try:
            self._hd5_file.close()
        except:
            pass
