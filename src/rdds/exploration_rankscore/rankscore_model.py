import numpy as np

from .dataset import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT


class GenmodRankScoreModel:
    """
    Implementation of thresholded rankscore model from Genmod.
    https://github.com/Clinical-Genomics/genmod/tree/master/genmod/score_variants

    The Genmod repository does not contain an actual _thresholded_ ranking model,
    but it does produce rank scores (inferences) based on annotated variant information.

    This class thresholds these rank score values into a binary classification problem,
    in order to allow computation of Genmod performance scores based on
    ground truth labels acquired from MUTACC database.
    """

    def __init__(self, threshold: float):
        self._threshold: float = threshold

    def predict(self, rank_scores: np.ndarray) -> np.ndarray:
        """
        Predicts variant class by means of converting rank_scores (-inf, +inf)
        to binary (0, 1) categorical predictions by applying a threshold.
        :param rank_scores:
        :return: np.ndarray as categorical predictions
        """
        preds: np.ndarray = np.where(rank_scores >= self._threshold,
                                     np.ones_like(rank_scores) * LABEL_PATHOGENIC_VARIANT,
                                     np.ones_like(rank_scores) * LABEL_BENIGN_VARIANT)
        return preds
