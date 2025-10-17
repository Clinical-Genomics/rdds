import numpy as np
from sklearn import metrics as sklearn_metrics
import copy
from typing import List
import matplotlib.pyplot as plt
import gc
import pandas as pd
from progressbar import ProgressBar

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


def plot_performance_vs_threshold(predictions: np.ndarray,
                                  labels: np.ndarray,
                                  output_path: str,
                                  figsize=_FIGSIZE,
                                  n_steps: int = 50):
    """
    Compute performance metrics at different thresholds
    :param predictions: Predictions (raw), not discretizised
    :param labels: Ground truth
    :param output_path: Image storage path, MUST contain .png suffix
    """
    f1_scores = []
    precision_scores = []
    recall_scores = []
    fnr_scores = []
    balanced_accuracy_scores = []
    mcc_scores = []
    thresholds = []
    pbar = ProgressBar(max_value=n_steps)
    pbar.start()
    for threshold in np.linspace(start=0, stop=1.0, num=n_steps):
        thresholds.append(threshold)
        disc_predictions = discretize_predictions(predictions=predictions,
                                                  threshold=threshold)
        f_score = sklearn_metrics.f1_score(y_true=labels,
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
        f1_scores.append(f_score)
        precision_scores.append(precision_score)
        recall_scores.append(recall_score)
        fnr_scores.append(score_fnr)
        balanced_accuracy_scores.append(bacc)
        mcc_scores.append(mcc_score)
        gc.collect()
        pbar.increment(1)
    pbar.finish()

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
    ax.plot(thresholds, f1_scores, marker='.')
    ax.plot(thresholds, mcc_scores, marker='.')
    ax.plot(thresholds, recall_scores, marker='.')
    ax.plot(thresholds, precision_scores, marker='.')
    ax.plot(thresholds, fnr_scores, marker='.')
    ax.plot(thresholds, balanced_accuracy_scores, marker='.')
    ax.legend([f'F1 {max_at_threshold(f1_scores, thresholds)}',
               f'MCC {max_at_threshold(mcc_scores, thresholds)}',
               f'Recall (sensitivity) {max_at_threshold(recall_scores, thresholds)}',
               f'Precision {max_at_threshold(precision_scores, thresholds)}',
               f'False Negative Rate (FNR) {min_at_threshold(fnr_scores, thresholds)}',
               f'Balanced Accuracy (BA) {max_at_threshold(balanced_accuracy_scores, thresholds)}'],
              loc='lower left')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score [F1, MCC, Recall, Precision, FNR, BA]')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.1)
    fig.suptitle('Performance scores vs discretization thresholds')
    fig.tight_layout()
    fig.savefig(output_path)
    df = pd.DataFrame(data={
        'thresholds': thresholds,
        'f1': f1_scores,
        'recall': recall_scores,
        'precision': precision_scores,
        'fnr': fnr_scores,
        'balanced_accuracy': balanced_accuracy_scores,
        'mcc': mcc_scores
    })
    df.to_csv(output_path.replace('.png', '.csv'))
    del fig, df
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