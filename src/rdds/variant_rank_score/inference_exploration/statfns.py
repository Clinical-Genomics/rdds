import os

import numpy as np
from sklearn import metrics as sklearn_metrics
import copy
from typing import List
import matplotlib.pyplot as plt
import gc
import pandas as pd
from progressbar import ProgressBar
from multiprocessing import Queue

from rdds.lib.process_pool import ProcessPool
from ..dataset.class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT

_FIGSIZE = (30, 20)

def discretize_predictions(predictions: np.ndarray,
                           threshold: float) -> np.ndarray:
    """
    Convert floating point predictions into binary labels by applying a threshold.
    :param predictions: Array containing floating point predictions
    :param threshold: The threshold used for converting floating point threshold to
    :return: Converted predictions array
    """
    predictions = copy.deepcopy(predictions)
    for i in range(0, len(predictions)):
        if predictions[i] >= threshold:
            predictions[i] = 1.0
        else:
            predictions[i] = 0.0
    return predictions


def fnr_score(predictions: np.ndarray,
               labels: np.ndarray) -> float:
    """
    Compute False Negative Rate (FNR)
    :param predictions: Model inferences (discretized)
    :param labels: Ground truth
    :return: FNR score
    """
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(y_true=labels,
                                                      y_pred=predictions).ravel()
    n_positives = 0
    for label in labels:
        if label == LABEL_PATHOGENIC_VARIANT:
            n_positives += 1
    if n_positives == 0:
        raise ValueError(f'Cannot compute FNR since no PATHOGENIC samples in data')
    return float(fn) / float(n_positives)


def _performance_vs_threshold_metrics_fn(*args, **kwargs):
    labels = kwargs['labels']
    predictions = kwargs['predictions']
    threshold = kwargs['threshold']
    result_queue = kwargs['result_queue']
    disc_predictions = discretize_predictions(predictions=predictions,
                                              threshold=threshold)
    f1_score = sklearn_metrics.f1_score(y_true=labels,
                                       y_pred=disc_predictions,
                                       pos_label=LABEL_PATHOGENIC_VARIANT)
    precision_score = sklearn_metrics.precision_score(y_true=labels,
                                                      y_pred=disc_predictions,
                                                      pos_label=LABEL_PATHOGENIC_VARIANT)
    recall_score = sklearn_metrics.recall_score(y_true=labels,
                                                y_pred=disc_predictions,
                                                pos_label=LABEL_PATHOGENIC_VARIANT)
    score_fnr = fnr_score(labels=labels,
                          predictions=disc_predictions)
    bacc = sklearn_metrics.balanced_accuracy_score(y_true=labels,
                                                   y_pred=disc_predictions,
                                                   adjusted=False)  # No need to scale to 1/n classes
    mcc_score = sklearn_metrics.matthews_corrcoef(y_true=labels,
                                                  y_pred=disc_predictions)

    result_queue.put({
        'f1_score': f1_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'score_fnr': score_fnr,
        'bacc': bacc,
        'mcc_score': mcc_score,
        'threshold': threshold
    })

def plot_performance_vs_threshold(predictions: np.ndarray,
                                  labels: np.ndarray,
                                  output_path: str,
                                  n_parallel_processes = int(os.cpu_count() / 2.0),
                                  figsize=_FIGSIZE,
                                  n_steps: int = 50):
    """
    Compute performance metrics at different thresholds
    :param predictions: Predictions (raw), not discretizised
    :param labels: Ground truth
    :param output_path: Image storage path, MUST contain .png suffix
    :param n_parallel_processes: Amount of workers in process pool to concurrently execute processing (impacts RAM usage)
    """

    result_queue: Queue = ProcessPool.get_context().Queue()

    f1_scores = []
    precision_scores = []
    recall_scores = []
    fnr_scores = []
    balanced_accuracy_scores = []
    mcc_scores = []
    thresholds = []
    pbar = ProgressBar(max_value=n_steps)
    pbar.start()

    kwargs = [{'predictions': predictions,
               'threshold': threshold,
               'labels': labels,
               'result_queue': result_queue} for threshold in np.linspace(start=0, stop=1.0, num=n_steps)]

    pool = ProcessPool(function=_performance_vs_threshold_metrics_fn,
                       kwargs=kwargs,
                       workers=n_parallel_processes)

    for task in pool.run():
        if task.process.exitcode != 0:
            raise ValueError(task)
        pbar.increment(1)
    pbar.finish()

    for _ in kwargs:
        result_dict = result_queue.get(timeout=60*60)
        thresholds.append(result_dict['threshold'])
        f1_scores.append(result_dict['f1_score'])
        precision_scores.append(result_dict['precision_score'])
        recall_scores.append(result_dict['recall_score'])
        fnr_scores.append(result_dict['score_fnr'])
        balanced_accuracy_scores.append(result_dict['bacc'])
        mcc_scores.append(result_dict['mcc_score'])

    result_df = pd.DataFrame({
        'thresholds': thresholds,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'fnr_scores': fnr_scores,
        'balanced_accuracy_scores': balanced_accuracy_scores,
        'mcc_scores': mcc_scores
    })
    result_df.sort_values(by='thresholds', inplace=True)


    def max_at_threshold(scores: list,
                         thresholds: list) -> str:
        idx = np.argmax(scores)
        return f'(max={scores[idx]:.4f}@{thresholds[idx]:.4f})'

    def min_at_threshold(scores: list,
                         thresholds: list) -> str:
        idx = np.argmin(scores)
        return f'(min={scores[idx]:.4f}@{thresholds[idx]:.4f})'

    fig: plt.Figure = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_subplot()
    ax.grid(True, which='both')
    ax.minorticks_on()
    ax.plot(result_df.thresholds, result_df.f1_scores, marker='.')
    ax.plot(result_df.thresholds, result_df.mcc_scores, marker='.')
    ax.plot(result_df.thresholds, result_df.recall_scores, marker='.')
    ax.plot(result_df.thresholds, result_df.precision_scores, marker='.')
    ax.plot(result_df.thresholds, result_df.fnr_scores, marker='.')
    ax.plot(result_df.thresholds, result_df.balanced_accuracy_scores, marker='.')
    ax.legend([f'F1 {max_at_threshold(result_df.f1_scores, result_df.thresholds)}',
               f'MCC {max_at_threshold(result_df.mcc_scores, result_df.thresholds)}',
               f'Recall (sensitivity) {max_at_threshold(result_df.recall_scores, result_df.thresholds)}',
               f'Precision {max_at_threshold(result_df.precision_scores, result_df.thresholds)}',
               f'False Negative Rate (FNR) {min_at_threshold(result_df.fnr_scores, result_df.thresholds)}',
               f'Balanced Accuracy (BA) {max_at_threshold(result_df.balanced_accuracy_scores, result_df.thresholds)}'],
              loc='lower left')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score [F1, MCC, Recall, Precision, FNR, BA]')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    fig.suptitle('Performance scores vs discretization thresholds')
    fig.tight_layout()
    fig.savefig(output_path)
    result_df.to_csv(output_path.replace('.png', '.csv'))
    del fig, result_df
    gc.collect()


def confusion_matrix(predictions: np.ndarray,
                     truths: np.ndarray,
                     discretisation_threshold: float,
                     output_path: str,
                     classes: List[float] = [LABEL_BENIGN_VARIANT, LABEL_PATHOGENIC_VARIANT],
                     figsize=_FIGSIZE):
    """
    Plot a confusion matrix for predictions thresholded at threshold
    :param predictions: Predictions (raw)
    :param truths: Ground truth labels
    :param discretisation_threshold: Threshold used to discretisize inferences
    :param output_path: Storage path for plot
    :param classes: List of classes
    """
    predictions = discretize_predictions(predictions, threshold=discretisation_threshold)
    cm = sklearn_metrics.confusion_matrix(y_true=truths,
                                          y_pred=predictions,
                                          labels=classes)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    cm_plot = sklearn_metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                     display_labels=classes)
    cm_plot.plot(ax=ax)
    fig.suptitle(f'Threshold={discretisation_threshold}')
    fig.tight_layout()
    fig.savefig(output_path)
    del fig
    gc.collect()


def plot_roc_auc(predictions: np.ndarray,
                 truths: np.ndarray,
                 output_path: str,
                 pos_label=LABEL_PATHOGENIC_VARIANT,
                 figsize=_FIGSIZE):
    """
    Compute and visualize ROC-AUC curve
    :param predictions: Inferences
    :param truths: Ground truth labels
    :param output_path: Figure storage path
    :param pos_label: Label of positive class
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    sklearn_metrics.RocCurveDisplay.from_predictions(y_true=truths,
                                                     y_pred=predictions,
                                                     pos_label=pos_label,
                                                     ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    del fig
    gc.collect()