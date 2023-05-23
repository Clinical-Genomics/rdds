from dataclasses import dataclass
import numpy as np

from .dataset import LABEL_BENIGN_VARIANT, LABEL_PATHOGENIC_VARIANT

_ALLOWED_LABELS = {LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT}


@dataclass
class ConfusionMatrix:
    true_positives: float
    true_negatives: float
    false_positives: float
    false_negatives: float


def as_confusion_matrix(predictions: np.ndarray,
                        labels: np.ndarray) -> ConfusionMatrix:
    """
    Compute the confusion matrix of predictions.

    Consider PATHOGENIC variants as the POSITIVE case.

    :param predictions: Predictions Z(0, 1)
    :param labels: Ground truth labels Z(0, 1)
    :return:
    """
    tps: float = 0.0  # True positives
    tns: float = 0.0  # True negatives
    fps: float = 0.0  # False positives
    fns: float = 0.0  # False negatives
    if not predictions.shape == labels.shape:
        raise ValueError('Bad predictions, labels shape')
    for prediction, label in zip(predictions, labels):
        if label not in _ALLOWED_LABELS:
            raise ValueError(f'Unknown label value: {label}')
        if prediction not in _ALLOWED_LABELS:
            raise ValueError(f'Unknown prediction value: {prediction}')
        if label == LABEL_PATHOGENIC_VARIANT:  # Positives
            if prediction == LABEL_BENIGN_VARIANT:
                fns += 1.0
            elif prediction == LABEL_PATHOGENIC_VARIANT:
                tps += 1.0
        else:  # Negatives
            if prediction == LABEL_BENIGN_VARIANT:
                tns += 1.0
            elif prediction == LABEL_PATHOGENIC_VARIANT:
                fps += 1.0
    return ConfusionMatrix(true_positives=tps,
                           true_negatives=tns,
                           false_positives=fps,
                           false_negatives=fns)


def add_confusion_matrices(a: ConfusionMatrix, b: ConfusionMatrix) -> ConfusionMatrix:
    """
    Adds two confusion matrixes, a and b, and returns the sum.
    :param a: Matrix a
    :param b: Matrix b
    :return: a + b Confusion Matrix
    """
    return ConfusionMatrix(true_positives=a.true_positives + b.true_positives,
                           true_negatives=a.true_negatives + b.true_negatives,
                           false_positives=a.false_positives + b.false_positives,
                           false_negatives=a.false_negatives + b.false_negatives)
