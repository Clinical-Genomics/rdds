import numpy as np

from rdds.exploration_rankscore.k_fold import get_k_fold


def test_k_fold():
    # GIVEN some data and labels
    # WHEN sampling subsets from these arrays
    n_samples = 1E3
    rank_scores: np.ndarray = np.arange(0, n_samples)
    labels: np.ndarray = np.arange(0, n_samples) * 0.1

    iters = 0
    for score, label in get_k_fold(rank_scores=rank_scores,
                                   labels=labels,
                                   subset_size_percentage=0.1):
        # THEN expect proper sizes and iterations
        assert len(score) == len(rank_scores) * 0.1, len(score)
        assert len(label) == len(labels) * 0.1, len(labels)
        iters += 1
    assert iters == 10

    iters = 0
    for score, label in get_k_fold(rank_scores=rank_scores,
                                   labels=labels,
                                   subset_size=100):
        # THEN expect proper sizes and iterations
        assert len(score) == 100, len(score)
        assert len(label) == 100, len(labels)
        iters += 1
    assert iters == 10
