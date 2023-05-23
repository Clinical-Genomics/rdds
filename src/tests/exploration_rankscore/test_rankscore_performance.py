import numpy as np
import pytest as pt
from math import isclose

from rdds.exploration_rankscore.dataset import LABEL_BENIGN_VARIANT, LABEL_PATHOGENIC_VARIANT
from rdds.exploration_rankscore.confusion_matrix import as_confusion_matrix, ConfusionMatrix
from rdds.exploration_rankscore.mcc_score import compute_mcc_score
from rdds.exploration_rankscore.f_score import compute_f_score
from rdds.exploration_rankscore.rankscore_performance import compute_optimal_performance_point


def test_confusion_matrix():
    # GIVEN some predictions, labels
    # WHEN computing the confusion matrix
    predictions: np.ndarray = np.array([0])
    labels: np.ndarray = np.array([0])
    matrix = as_confusion_matrix(predictions=predictions, labels=labels)
    # THEN expect TN result only
    assert matrix.true_positives == 0
    assert matrix.true_negatives == 1
    assert matrix.false_negatives == 0
    assert matrix.false_positives == 0

    predictions: np.ndarray = np.array([1])
    labels: np.ndarray = np.array([1])
    matrix = as_confusion_matrix(predictions=predictions, labels=labels)
    # THEN expect TP result only
    assert matrix.true_positives == 1
    assert matrix.true_negatives == 0
    assert matrix.false_negatives == 0
    assert matrix.false_positives == 0

    predictions: np.ndarray = np.array([0])
    labels: np.ndarray = np.array([1])
    matrix = as_confusion_matrix(predictions=predictions, labels=labels)
    # THEN expect FN result only
    assert matrix.true_positives == 0
    assert matrix.true_negatives == 0
    assert matrix.false_negatives == 1
    assert matrix.false_positives == 0

    predictions: np.ndarray = np.array([1])
    labels: np.ndarray = np.array([0])
    matrix = as_confusion_matrix(predictions=predictions, labels=labels)
    # THEN expect FP result only
    assert matrix.true_positives == 0
    assert matrix.true_negatives == 0
    assert matrix.false_negatives == 0
    assert matrix.false_positives == 1


def test_mcc_score():
    """
    Test MCC score for different confusion matrix cases.
    :return:
    """
    # GIVEN a confusion matrix
    # WHEN computing MCC score
    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=1,
                                              false_positives=0,
                                              false_negatives=0)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (ideal)
    assert isclose(score, 1.0), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=0,
                                              false_positives=0,
                                              false_negatives=0)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (no TNs found)
    assert isclose(score, 0.75, abs_tol=1E-6), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=1,
                                              false_positives=0,
                                              false_negatives=0)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (no TPs found)
    assert isclose(score, 0.75, abs_tol=1E-6), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_positives=1,
                                              false_negatives=0)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (at least no FNs)
    assert isclose(score, 0.25, abs_tol=1E-6), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_positives=0,
                                              false_negatives=1)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (at least no FPs)
    assert isclose(score, 0.25, abs_tol=1E-6), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_positives=1,
                                              false_negatives=1)
    score = compute_mcc_score(confusion_matrix=matrix)
    # EXPECT valid result (complete disagreement)
    assert isclose(score, 0.0), score


def test_f_score():
    """
    Test for F(1) performance score.
    :return:
    """
    # GIVEN some confusion matrix
    # WHEN computing the performance score
    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=0,
                                              false_negatives=0,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix
    score = compute_f_score(confusion_matrix=matrix)
    assert score == 1.0, score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=1,
                                              false_negatives=0,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix (TN limitation case 1)
    score = compute_f_score(confusion_matrix=matrix)
    assert score == 1.0, score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=1,
                                              false_negatives=0,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix (TN limitation case 2)
    score = compute_f_score(confusion_matrix=matrix)
    assert score == 0.0, score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=1,
                                              false_negatives=0,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix
    score = compute_f_score(confusion_matrix=matrix)
    assert score == 0.0, score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=0,
                                              false_negatives=1,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix
    score = compute_f_score(confusion_matrix=matrix)
    assert isclose(score, 2.0/3.0), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=0,
                                              false_negatives=1,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix (ignore FNs)
    score = compute_f_score(confusion_matrix=matrix, beta=0.0)
    assert score == 1.0, score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=0,
                                              false_negatives=0,
                                              false_positives=1)
    # THEN expect this to match the confusion matrix
    score = compute_f_score(confusion_matrix=matrix)
    assert isclose(score, 2.0/3.0), score

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_negatives=1,
                                              false_positives=0)
    # THEN expect this to match the confusion matrix
    score = compute_f_score(confusion_matrix=matrix)
    assert score == 0.0, score


def test_performance_statistics():
    """
    Tests overall performance metric accuracy.
    :return:
    """
    # GIVEN some predictions and ground truth
    # WHEN computing optimal point of performance
    rank_scores: np.ndarray = np.array([10, 15])
    labels: np.ndarray = np.array([LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT])  # All TP case
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=1,
                                                threshold_steps=100,
                                                save_plots=False)
    # THEN expect it to be at optimal region
    assert metrics.mean_best_operating_threshold < 10
    assert metrics.roc_auc == 0.0  # There are no TNs in this dataset, so area is 0
    assert isclose(metrics.mean_score_at_operating_point, 0.75, abs_tol=1E-3)

    rank_scores: np.ndarray = np.array([-10, -2])
    labels: np.ndarray = np.array([LABEL_BENIGN_VARIANT, LABEL_BENIGN_VARIANT])  # All TN case
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=1,
                                                threshold_steps=100,
                                                save_plots=False)
    assert metrics.mean_best_operating_threshold > -2
    assert metrics.roc_auc == 0.0
    assert isclose(metrics.mean_score_at_operating_point, 0.75, abs_tol=1E-3)

    rank_scores: np.ndarray = np.array([-10, -2, 0, 1, 2, 3])
    labels: np.ndarray = np.array([LABEL_BENIGN_VARIANT, LABEL_BENIGN_VARIANT,
                                   LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT,
                                   LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT])  # Mixed TN, TP case
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=len(rank_scores),
                                                threshold_steps=100,
                                                save_plots=False)
    assert -2 < metrics.mean_best_operating_threshold < 0
    assert metrics.roc_auc == 1.0
    assert metrics.mean_score_at_operating_point == 1.0

    rank_scores: np.ndarray = np.array([1, -5, -1, 2, -10])
    labels: np.ndarray = np.array([LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT,
                                   LABEL_BENIGN_VARIANT, LABEL_PATHOGENIC_VARIANT,
                                   LABEL_BENIGN_VARIANT])  # Mixed TN, TP case
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=2,
                                                threshold_steps=100,
                                                save_plots=False)
    assert -1 < metrics.mean_best_operating_threshold < 1
    assert metrics.roc_auc == 1.0
    assert metrics.mean_score_at_operating_point == 1.0

    rank_scores: np.ndarray = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    labels: np.ndarray = np.array([LABEL_BENIGN_VARIANT, LABEL_BENIGN_VARIANT,
                                   LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT,
                                   LABEL_BENIGN_VARIANT, LABEL_BENIGN_VARIANT,
                                   LABEL_BENIGN_VARIANT, LABEL_PATHOGENIC_VARIANT,
                                   LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT])  # Two regions; 2*FN + 3*TP, 5*TN
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=len(rank_scores),
                                                threshold_steps=100,
                                                save_plots=False)
    assert 6 < metrics.mean_best_operating_threshold < 10
    assert isclose(metrics.roc_auc, 0.76, abs_tol=1E-3)
    assert isclose(metrics.mean_score_at_operating_point, 0.827, abs_tol=1E-3)

    rank_scores = (rank_scores - 0) / (9 - 0)  # maxmin normalized rank score
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=len(rank_scores),
                                                threshold_steps=100,
                                                save_plots=False)
    assert 6.0/9.0 < metrics.mean_best_operating_threshold < 10.0/9.0
    assert isclose(metrics.roc_auc, 0.76, abs_tol=1E-3)
    assert isclose(metrics.mean_score_at_operating_point, 0.827, abs_tol=1E-3)

    rank_scores: np.ndarray = np.array([0, 1, 2, 3])
    labels: np.ndarray = np.array([LABEL_PATHOGENIC_VARIANT, LABEL_PATHOGENIC_VARIANT,
                                   LABEL_BENIGN_VARIANT, LABEL_BENIGN_VARIANT])  #  Separation not possible
    metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                labels=labels,
                                                k_fold_subset_size=2,
                                                threshold_steps=100,
                                                save_plots=False)
    assert 0 <= metrics.mean_best_operating_threshold <= 3
    assert metrics.roc_auc == 0.0
    assert metrics.mean_score_at_operating_point == 0.5
