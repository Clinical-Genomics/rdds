from copy import copy
from dataclasses import asdict
import numpy as np

from .confusion_matrix import ConfusionMatrix


def compute_mcc_score(confusion_matrix: ConfusionMatrix,
                      beta: float = 1E-6) -> float:
    """
    Computes normalized Matthews correlation coefficient (MCC) score in range (0, 1).
    :param confusion_matrix: confusion matrix
    :param beta: Small value for numerical stability during zero-division case
    :return: MCC score in range (0, 1)
    """
    # https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7
    # https://en.wikipedia.org/wiki/Phi_coefficient

    def compute_mcc(confusion_matrix: ConfusionMatrix) -> float:
        numerator: float = (confusion_matrix.true_positives * confusion_matrix.true_negatives) - \
                           (confusion_matrix.false_positives * confusion_matrix.false_negatives)
        denominator: float = (confusion_matrix.true_positives + confusion_matrix.false_positives) * \
                             (confusion_matrix.true_positives + confusion_matrix.false_negatives) * \
                             (confusion_matrix.true_negatives + confusion_matrix.false_positives) * \
                             (confusion_matrix.true_negatives + confusion_matrix.false_negatives)
        denominator = np.sqrt(denominator)
        return float(numerator) / float(denominator)  # Forcing types necessary for raising ZeroDivisionError

    confusion_matrix = confusion_matrix
    try:
        mcc = compute_mcc(confusion_matrix)
    except ZeroDivisionError:
        # Replace zeros with a small value, beta for numerical stability
        confusion_matrix_copy = copy(confusion_matrix)
        for key, value in asdict(confusion_matrix_copy).items():
            if value == 0.0:
                confusion_matrix_copy.__setattr__(key, beta)
        mcc = compute_mcc(confusion_matrix_copy)
    # mcc is normally in the range of (-1, +1)
    mcc_normalized: float = (mcc + 1.0) / 2.0
    return mcc_normalized
