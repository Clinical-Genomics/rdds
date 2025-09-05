import os
from h5py import File as Hdf5File, string_dtype
from typing import Set, Dict, Any
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import gc

from ..dataset.class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT
from .. import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.list_dir import list_dir
import rdds.variant_rank_score.inference_exploration.statfns as statfns

_LOGGER = get_logger(name='inference_explorer',
                     log_level='debug')

FIGSIZE = (30, 20)

# TODO: Stratify labels based on data properties
# TODO: Sort and view variants with worst residuals!
#   - Scatter plot (label, residual, feature)


class LabelClass(Enum):
    TRUE_POSITIVE = 'true-positive'
    TRUE_NEGATIVE = 'true-negative'
    FALSE_NEGATIVE = 'false-negative'
    FALSE_POSITIVE = 'false-positive'


class InferenceExplorer:

    """
    Visualize model inferences w.r.t. ground truth labels.
    """

    def __init__(self,
                 hd5_file_path: str,
                 ground_truth_column_name: str = 'ground-truth',
                 inferences_column_name: str = 'prediction',
                 groups={'train', 'test'},
                 output_dir=os.path.join(WORKDIR, 'inference_viz'),
                 inference_discretization_threshold: float = 0.5):
        """

        :param hd5_file_path: File path containing data and model inferences
        :param ground_truth_column_name: The name of the ground truth data set
        :param inferences_column_name: The name of the model inferences data set
        :param groups: The groups to analyze
        :param output_dir: The storage directory for module output
        :param inference_discretization_threshold: Threshold to use for turning inferences to binary predictions
        """
        self._hd5_file_path: str = hd5_file_path
        self._ground_truth_column_name = ground_truth_column_name
        self._inferences_column_name = inferences_column_name
        self._threshold: float = inference_discretization_threshold
        _LOGGER.info(f'Input file: {self._hd5_file_path}')
        self._groups: Set[str] = groups

        # Create/ clean workdir
        self._output_dir = output_dir
        try:
            os.mkdir(self._output_dir)
        except (FileExistsError):
            files = list(list_dir(directory_path=self._output_dir))
            [os.remove(file_name) for file_name in files]
            _LOGGER.info(f'Removed stale files in {self._output_dir}')
        _LOGGER.info(f'Storing output in {self._output_dir}')
        self._data = self._load_hd5_to_ram(hd5_file_path=hd5_file_path,
                                           groups=self._groups)

    @staticmethod
    def _load_hd5_to_ram(hd5_file_path: str,
                         groups: Set[str]) -> Dict[str,  pd.DataFrame]:
        """
        Load HD5 file contents containing groups to RAM
        :param hd5_file_path: Path to hd5 file
        :param groups: Groups to load
        :return: A dict mapping group_name -> DataFrame
        """
        # Load data to RAM
        data: Dict[str, Dict[str, Any]] = dict()
        hdf5_file = Hdf5File(name=hd5_file_path, mode='r')
        _LOGGER.debug('Loading data to RAM')
        for group in groups:
            feature_names: Set[str] = set()
            feature_names.update(set(list(hdf5_file[group].keys())))
            group_data: Dict[str, Any] = dict()
            for feature in feature_names:
                _LOGGER.debug(f'Loading {group}/{feature}')
                group_data.update({feature: hdf5_file[group][feature][:]})
            data.update({group: group_data})
        hdf5_file.close()

        dataframes: Dict[str, pd.DataFrame] = dict()
        for group_name in data.keys():
            dataframes.update({group_name: pd.DataFrame(data=data[group_name])})
        return dataframes

    @staticmethod
    def _compute_inference_label_class(prediction: int,
                                       label: int) -> str:
        """
        Compute inference category as TP, TN, FP, FN
        :param prediction: Thresholded prediction as integer
        :param label: Ground truth label
        :return: string containing the category
        """
        if label == LABEL_BENIGN_VARIANT:
            if prediction == LABEL_BENIGN_VARIANT:
                return LabelClass.TRUE_NEGATIVE.value
            elif prediction == LABEL_PATHOGENIC_VARIANT:
                return LabelClass.FALSE_POSITIVE.value
            else:
                raise ValueError(f'Unknown prediction {prediction}')
        elif label == LABEL_PATHOGENIC_VARIANT:
            if prediction == LABEL_PATHOGENIC_VARIANT:
                return LabelClass.TRUE_POSITIVE.value
            elif prediction == LABEL_BENIGN_VARIANT:
                return LabelClass.FALSE_NEGATIVE.value
            else:
                raise ValueError(f'Unknown prediction {prediction}')
        else:
            raise ValueError(f'Unknown label {label}')

    def _plot_scores_vs_threshold(self):
        for group_name in self._data.keys():
            predictions = self._data[group_name][self._inferences_column_name].values
            labels = self._data[group_name][self._ground_truth_column_name].values
            output_path = os.path.join(self._output_dir, f'performance-vs-thresholds-{group_name}.png')
            statfns.plot_performance_vs_threshold(predictions=predictions,
                                                  labels=labels,
                                                  output_path=output_path)

    def _plot_roc_auc(self):
        """
        Compute and visualize ROC-AUC curve
        """
        for group_name in self._data.keys():
            predictions_raw = self._data[group_name][self._inferences_column_name].values
            labels = self._data[group_name][self._ground_truth_column_name].values
            statfns.plot_roc_auc(predictions=predictions_raw,
                                 truths=labels,
                                 output_path=os.path.join(self._output_dir, f'roc-auc-{group_name}.png'))

    def _plot_residuals(self):
        """
        Visualize residuals as violin and plot.
        """
        for group_name, group in self._data.items():
            residuals = self._data[group_name][self._ground_truth_column_name] - self._data[group_name][self._inferences_column_name]
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.add_subplot()
            sb.violinplot(residuals, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(self._output_dir, f'residuals-{group_name}.png'))

            fig = plt.figure(figsize=FIGSIZE)
            ax: plt.Axes = fig.add_subplot()
            ax.scatter(x=self._data[group_name][self._ground_truth_column_name].values,
                       y=residuals.values,
                       color="blue",
                       alpha=0.5,
                       marker='.')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Residuals')
            ax.axhline(y=0, color="r", linestyle="-")
            ax.legend(['Variant'])
            fig.tight_layout()
            fig.savefig(os.path.join(self._output_dir, f'residuals-scatter-{group_name}.png'))
            del fig
        gc.collect()

    def _confusion_matrix(self):
        """
        Plot a confusion matrix for predictions thresholded at threshold
        """
        for group_name in self._data.keys():
            predictions = statfns.discretize_predictions(self._data[group_name][self._inferences_column_name].values,
                                                         threshold=self._threshold)
            y_true = self._data[group_name][self._ground_truth_column_name].values
            statfns.confusion_matrix(predictions=predictions,
                                     truths=y_true,
                                     discretisation_threshold=self._threshold,
                                     output_path=os.path.join(self._output_dir, f'confusion-{group_name}.png'))

    def _dump_prediction_class_to_hd5(self) -> str:
        """
        Store FP, FN samples to file for manual inspection.
        """

        _LOGGER.debug(f'Dumping prediction classes with threshold {self._threshold}')
        # Copy data from input dataset to a new file and append the prediction class
        ref_hd5_file = Hdf5File(self._hd5_file_path, 'r')
        file_name = 'prediction-analytics-'+os.path.basename(self._hd5_file_path)
        file_path = os.path.join(self._output_dir, file_name)
        analytics_hd5_file = Hdf5File(file_path, 'w')
        for group_name in self._data.keys():
            group = analytics_hd5_file.create_group(group_name)
            for feature_name in self._data[group_name]:
                shape = ref_hd5_file[group_name][feature_name].shape
                dtype = ref_hd5_file[group_name][feature_name].dtype
                group.create_dataset(feature_name,
                                     shape=shape,
                                     dtype=dtype,
                                     data=ref_hd5_file[group_name][feature_name][:])
            # Create a dataset for str data type, containing the prediction class
            group.create_dataset('prediction-class',
                                 shape=shape,
                                 dtype=string_dtype(),
                                 fillvalue=b'\0')
            prediction_class = group['prediction-class'][:]  # Alloc in RAM

            predictions = statfns.discretize_predictions(self._data[group_name][self._inferences_column_name].values,
                                                         threshold=self._threshold)
            labels = self._data[group_name][self._ground_truth_column_name].values
            for i in range(0, len(predictions)):
                prediction_class[i]: str = self._compute_inference_label_class(prediction=predictions[i],
                                                                               label=labels[i])
            group['prediction-class'][:] = prediction_class
        ref_hd5_file.close()
        analytics_hd5_file.flush()
        analytics_hd5_file.close()
        gc.collect()
        return file_path

    def _hd5_to_tsv(self,
                    hd5_file_path: str,
                    groups: Set[str]):
        data = self._load_hd5_to_ram(hd5_file_path,
                                     groups=groups)
        for group_name in data.keys():
            output_file_name = os.path.basename(hd5_file_path)+f'-{group_name}.tsv'
            with open(os.path.join(self._output_dir, output_file_name), 'w') as file:
                feature_names = data[group_name].keys()
                [file.write(f'{feature_name}\t') for feature_name in feature_names]
                file.write('\n')
                data_length = len(data[group_name][feature_names[-1]])
                string_chunk: str = ''
                for row_idx in range(0, data_length):
                    for feature_name in feature_names:
                        string_chunk += f'{data[group_name].iloc[row_idx][feature_name]}\t'
                    string_chunk += '\n'
                    if row_idx % 1E4 == 0:
                        file.write(string_chunk)
                        string_chunk = ''
                if len(string_chunk) > 0:
                    file.write(string_chunk)

    def __call__(self):
        self._confusion_matrix()
        self._plot_residuals()
        self._plot_scores_vs_threshold()
        self._plot_roc_auc()
        prediction_classes_hd5_file = self._dump_prediction_class_to_hd5()
        self._hd5_to_tsv(hd5_file_path=prediction_classes_hd5_file,
                         groups=self._groups)

