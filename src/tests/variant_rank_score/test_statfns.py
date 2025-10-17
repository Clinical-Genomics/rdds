import numpy as np
from os.path import join

from rdds.variant_rank_score.inference_exploration.statfns import plot_performance_vs_threshold



def test_performance_vs_threshold(work_dir):
    """
    Test for configuring plotting behavior
    """
    v = np.linspace(0, 1, 20)
    l = np.zeros(20)
    l[10:] = 1
    plot_performance_vs_threshold(predictions=v,
                                  labels=l,
                                  output_path=join(work_dir, 'p.png'))