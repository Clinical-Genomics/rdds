from dataclasses import dataclass

from .confusion_matrix import ConfusionMatrix


@dataclass
class RocMetricCoordinate:
    false_positive_rate: float  # FPR, x axis
    true_positive_rate: float  # TPR, y axis


def compute_roc(confusion_matrix: ConfusionMatrix) -> RocMetricCoordinate:
    """
    Compute ROC metric from confusion matrix.
    :param confusion_matrix:
    :return: RocMetricCoordinate object
    """
    try:
        false_positive_rate = confusion_matrix.false_positives / \
                              (confusion_matrix.false_positives + confusion_matrix.true_negatives)
    except ZeroDivisionError:
        false_positive_rate: float = 0.0
    try:
        true_positive_rate = confusion_matrix.true_positives / \
                             (confusion_matrix.true_positives + confusion_matrix.false_negatives)
    except ZeroDivisionError:
        true_positive_rate: float = 0.0

    return RocMetricCoordinate(false_positive_rate=false_positive_rate, true_positive_rate=true_positive_rate)
