import numpy as np
from os.path import join
import pytest as pt

from rdds.variant_rank_score.inference_exploration.statfns import plot_performance_vs_threshold


@pt.mark.parametrize('n_samples', [20, int(1E6)])
def test_performance_vs_threshold(work_dir, n_samples):
    """
    Test for configuring plotting behavior
    """
    v = np.linspace(0, 1, n_samples)
    l = np.zeros_like(v)
    l[int(n_samples/2):] = 1
    plot_performance_vs_threshold(predictions=v,
                                  labels=l,
                                  output_path=join(work_dir, 'p.png'))