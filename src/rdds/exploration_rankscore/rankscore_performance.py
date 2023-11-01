import numpy as np
from typing import *
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from . import WORKDIR
_DPI = 300
from .rankscore_model import GenmodRankScoreModel
from .k_fold import get_k_fold
from .confusion_matrix import as_confusion_matrix, add_confusion_matrices, ConfusionMatrix
from .mcc_score import compute_mcc_score
from .f_score import compute_f_score
from .roc import compute_roc

"""
Computes rank score performance metrics with the help of ground truth(True Positive, TP)
causative variants.

Visualizes rank score model performance by means of MCC and F score across thresholds.
"""


@dataclass
class ThresholdPerformanceResult:
    threshold: float
    mcc_scores: List[float]
    f_scores: List[float]
    confusion_matrix: ConfusionMatrix


@dataclass
class PerformanceResult:
    thresholds: np.ndarray
    mcc_scores: np.ndarray
    mcc_auc: float
    f_scores: np.ndarray
    f_score_auc: float
    roc_auc: float
    best_operating_thresholds: np.ndarray
    scores_at_best_operating_point: np.ndarray
    mean_best_operating_threshold: float
    mean_score_at_operating_point: float


def compute_optimal_performance_point(rank_scores: np.ndarray,
                                      labels: np.ndarray,
                                      threshold_steps: int = 100,
                                      save_plots: bool = False,
                                      plot: bool = False,
                                      k_fold_subset_size: int = None,
                                      main_evaluation_metric: str = 'mcc_score',
                                      image_name_prefix: str = None) -> PerformanceResult:
    """
    Compute performance metrics for Rank Score Model and return the optimal performance point
    with respect to rank score threshold.

    NOTE: Be aware that k_fold_subset_size parameter affects the performance scores.
    Make sure that the fold has enough samples to cover all confusion matrix cases for the most
    accurate evaluation.

    # TODO: Provide info on confusion matrix at best point, merge with variant IDs for deeper investigation capabilities

    :param rank_scores: Array of rank scores
    :param labels: Ground truth labels (binary, categorical)
    :param threshold_steps: Number of steps to sweep threshold
    :param save_plots: Save plots to png files in working directory
    :param plot: Show plots interactively
    :param k_fold_subset_size: Size of fold for cross-validation
    :param main_evaluation_metric: The main metric to select best operation point from. [mcc_score|f_score]
    :param image_name_prefix: Prefix to add to saved image names. Str
    :return: Metrics object
    """
    if not len(rank_scores.shape) == 1 or not len(labels.shape) == 1:
        raise TypeError('Only 1D arrays supported')
    if not len(rank_scores) == len(labels):
        raise ValueError(f'Mismatch length data and labels {rank_scores.shape}{labels.shape}')

    image_name_prefix = '' if image_name_prefix is None else image_name_prefix + '-'

    # Generate rankscore model performance data by using sweeping threshold
    threshold_min: float = np.min(rank_scores)
    threshold_max: float = np.max(rank_scores) * 1.1
    threshold_magnitude = threshold_max - threshold_min
    threshold_min -= threshold_magnitude * 0.1  # Add threshold padding
    threshold_max += threshold_magnitude * 0.1
    thresholds: np.ndarray = np.linspace(threshold_min, threshold_max, threshold_steps)
    performance_results: List[ThresholdPerformanceResult] = []
    for threshold in thresholds:
        mcc_scores: List[float] = []
        f_scores: List[float] = []
        total_confusion_matrix: ConfusionMatrix = None
        for subset_rank_scores, subset_labels in get_k_fold(rank_scores=rank_scores,
                                                             labels=labels,
                                                             subset_size=k_fold_subset_size):
            predictions = GenmodRankScoreModel(threshold=threshold).predict(rank_scores=subset_rank_scores)
            confusion_matrix = as_confusion_matrix(predictions, subset_labels)
            mcc_score = compute_mcc_score(confusion_matrix=confusion_matrix)
            mcc_scores.append(mcc_score)
            f_score = compute_f_score(confusion_matrix=confusion_matrix)
            f_scores.append(f_score)
            total_confusion_matrix = confusion_matrix if total_confusion_matrix is None \
            else add_confusion_matrices(total_confusion_matrix, confusion_matrix)
        performance_results.append(ThresholdPerformanceResult(threshold=threshold,
                                                              mcc_scores=mcc_scores,
                                                              f_scores=f_scores,
                                                              confusion_matrix=total_confusion_matrix))
    # Compute various metrics across all thresholds
    mean_mcc_scores = np.array([np.mean(result.mcc_scores) for result in performance_results])
    mean_f_scores = np.array([np.mean(result.f_scores) for result in performance_results])
    std_mcc_scores = np.array([np.std(result.mcc_scores) for result in performance_results])
    std_f_scores = np.array([np.std(result.f_scores) for result in performance_results])
    max_mcc_score: np.float = np.max(mean_mcc_scores)
    max_f_score: np.float = np.max(mean_f_scores)
    # Depending on the threshold resolution (delta spacing) there can be multiple MAX(score) indexes
    max_mcc_score_idxs: np.ndarray = np.argwhere(mean_mcc_scores == max_mcc_score)[:, 0]
    max_f_score_idxs: np.ndarray = np.argwhere(mean_f_scores == max_f_score)[:, 0]
    best_mcc_scores: np.ndarray = mean_mcc_scores[max_mcc_score_idxs]
    best_f_scores: np.ndarray = mean_f_scores[max_f_score_idxs]
    best_mcc_thresholds: np.ndarray = thresholds[max_mcc_score_idxs]
    best_f_thresholds: np.ndarray = thresholds[max_f_score_idxs]
    auc_mcc: float = auc(thresholds, mean_mcc_scores)
    auc_f_score: float = auc(thresholds, mean_f_scores)
    false_positive_rates: np.ndarray = np.zeros(len(performance_results))
    true_positive_rates: np.ndarray = np.zeros(len(performance_results))
    for idx, result in enumerate(performance_results):
        roc_coordinate = compute_roc(result.confusion_matrix)
        false_positive_rates[idx] = roc_coordinate.false_positive_rate
        true_positive_rates[idx] = roc_coordinate.true_positive_rate
    auc_roc: float = auc(false_positive_rates, true_positive_rates)
    n_folds = len(performance_results[0].mcc_scores)

    # Plotting performance scores
    # Plot MCC score
    figure = plt.figure(figsize=(15, 15))
    axis = figure.add_subplot(111)
    axis.plot(thresholds, mean_mcc_scores, color='black', label='Mean MCC Score across folds\n(AUC=%.2f)' % auc_mcc)
    for n_fold in range(0, n_folds):
        mcc_scores_at_fold = [result.mcc_scores[n_fold] for result in performance_results]
        axis.plot(thresholds, mcc_scores_at_fold, '--', alpha=0.75, label='MCC-at-fold-%d' % n_fold)
    axis.fill_between(thresholds,
                      np.add(mean_mcc_scores, std_mcc_scores),
                      np.subtract(mean_mcc_scores, std_mcc_scores),
                      color="grey",
                      alpha=0.2,
                      label=r"$\pm$ 1 std. dev.")
    axis.set_ylim(-0.1, 1.1)
    axis.set_xlabel('Threshold')
    axis.set_ylabel('MCC Score')
    axis.legend(loc='lower right')
    plt.suptitle('MCC Score')
    if save_plots:
        figure.savefig(WORKDIR + f'/{image_name_prefix}mcc-score.png', dpi=_DPI)

    # Plot F1 score
    figure = plt.figure(figsize=(15, 15))
    axis = figure.add_subplot(111)
    axis.plot(thresholds, mean_f_scores, color='black', label='Mean F Score across folds\n(AUC=%.2f)' % auc_f_score)
    for n_fold in range(0, n_folds):
        f_score_at_fold = [result.f_scores[n_fold] for result in performance_results]
        axis.plot(thresholds, f_score_at_fold, '--', alpha=0.75, label='F-score-fold-%d' % n_fold)
    axis.fill_between(thresholds,
                      np.add(mean_f_scores, std_f_scores),
                      np.subtract(mean_f_scores, std_f_scores),
                      color="grey",
                      alpha=0.2,
                      label=r"$\pm$ 1 std. dev.")
    axis.set_ylim(-0.1, 1.1)
    axis.set_xlabel('Threshold')
    axis.set_ylabel('F Score')
    axis.legend(loc='lower right')
    plt.suptitle('F Score')
    if save_plots:
        figure.savefig(WORKDIR + f'/{image_name_prefix}f-score.png', dpi=_DPI)

    # Plot Total Operating Characteristic (TOC)
    # https://doi.org/10.1080/13658816.2013.862623
    # https://www.mdpi.com/2072-4292/13/19/3922
    # https://en.wikipedia.org/wiki/Total_operating_characteristic#
    hits: np.ndarray = np.array([result.confusion_matrix.true_positives for result in performance_results])
    false_alarms: np.ndarray = np.array([result.confusion_matrix.false_positives for result in performance_results])
    correct_rejections: np.ndarray = np.array([result.confusion_matrix.true_negatives for result in performance_results])
    toc_x_axis: np.ndarray = np.add(hits, false_alarms)
    figure = plt.figure(figsize=(15, 15))
    axis = figure.add_subplot(111)
    axis.plot(toc_x_axis, hits, label='Hits', color='black')
    # TODO: Annotate points with threshold values, currently clobber itself
    #for idx, (x, y) in enumerate(zip(toc_x_axis, hits)):
    #    axis.annotate('%.2f' % thresholds[idx], xy=(x+.01, y+.01))
    axis.plot(toc_x_axis - false_alarms, hits, '--', label='Minimum', color='blue')
    axis.plot(toc_x_axis + correct_rejections, hits, '--', label='Maximum', color='red')
    axis.plot(np.linspace(min(toc_x_axis), max(toc_x_axis), len(toc_x_axis)),
              np.linspace(min(hits), max(hits), len(toc_x_axis)), '--', label='Uniform, random classifier', color='black', alpha=0.5)
    axis.set_xlabel('Hits + False Alarms')
    axis.set_ylabel('Hits')
    axis.legend(loc='lower right')
    plt.suptitle('Total Operating Characteristic (TOC)\nAnnotation values in graph correspond to threshold in effect.')
    if save_plots:
        figure.savefig(WORKDIR + f'/{image_name_prefix}toc.png', dpi=_DPI)

    # Plot ROC-AUC
    figure = plt.figure(figsize=(15, 15))
    axis = figure.add_subplot(111)
    axis.plot(false_positive_rates, true_positive_rates, label='ROC\n(AUC=%.2f)' % auc_roc, color='black')
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.legend(loc='lower right')
    plt.suptitle('Receiver Operating Characteristic (ROC)')
    if save_plots:
        figure.savefig(WORKDIR + f'/{image_name_prefix}roc.png', dpi=_DPI)

    if plot:
        plt.show()
    plt.close('all')

    if main_evaluation_metric == 'mcc_score':
        best_thresholds = best_mcc_thresholds
        best_scores = best_mcc_scores
    elif main_evaluation_metric == 'f_score':
        best_thresholds = best_f_thresholds
        best_scores = best_f_scores
    else:
        raise ValueError(f'Unknown main performance metric {main_evaluation_metric}')

    return PerformanceResult(thresholds=thresholds,
                             mcc_scores=mean_mcc_scores,
                             mcc_auc=auc_mcc,
                             f_scores=mean_f_scores,
                             f_score_auc=auc_f_score,
                             best_operating_thresholds=best_thresholds,
                             scores_at_best_operating_point=best_scores,
                             mean_best_operating_threshold=float(np.mean(best_thresholds)),
                             mean_score_at_operating_point=float(np.mean(best_scores)),
                             roc_auc=auc_roc)
