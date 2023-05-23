from .confusion_matrix import ConfusionMatrix


def compute_f_score(confusion_matrix: ConfusionMatrix,
                    beta: float = 1.0) -> float:
    """
    Compute F-score.
    When beta equals to one, then compute the balanced precision-recall
    F score (harmonic mean).

    NOTE: When matrix contains all True Negatives, this metric is limited
    and won't perform well. This is also the case in a very class imbalanced
    data set. Use MCC score as workaround.

    :param confusion_matrix: Confusion matrix
    :param beta: Weighting parameter (increasing beta favors recall)
    :return: F score in range (0, 1)
    """
    precision: float = 0.0
    recall: float = 0.0
    if confusion_matrix.true_positives > 0 or confusion_matrix.false_positives > 0:
        precision: float = confusion_matrix.true_positives / \
                       (confusion_matrix.true_positives + confusion_matrix.false_positives)
    if confusion_matrix.true_positives > 0 or confusion_matrix.false_negatives > 0:
        recall: float = confusion_matrix.true_positives / \
                    (confusion_matrix.true_positives + confusion_matrix.false_negatives)
    if precision + recall == 0:
        # Corner case when all TNs.
        return 0.0
    return (1.0 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
