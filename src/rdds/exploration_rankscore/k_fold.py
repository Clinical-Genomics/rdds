import numpy as np


def get_k_fold(rank_scores: np.ndarray,
                labels: np.ndarray,
                subset_size: int = None,
                subset_size_percentage: float = 0.1):
    """
    Return subset of data for k-fold sampling (tumbling).
    Don't use scikit methods here since we favor reproducibility of splits.

    :param rank_scores: rank scores in 1D array
    :param labels: categorical labels in 1D array
    :param subset_size: Desired k fold size, 10% of data length if not defined.
      If subset_size equals to data length, all data will be collected in 1st fold.
    :return: Iterator, subset of rank_scores, labels
    """

    if not len(rank_scores) == len(labels):
        raise ValueError('Mismatch length data and labels')
    dlen: int = len(rank_scores)
    if subset_size is None:
        subset_size: int = int(np.floor(dlen * subset_size_percentage))
    if subset_size == 0:
        raise ValueError('Subset size equals to zero, data too short')
    elif subset_size >= dlen:
        yield rank_scores, labels
        return
    for idx in range(subset_size, dlen + 1, subset_size):
        yield rank_scores[idx-subset_size:idx], labels[idx-subset_size:idx]
