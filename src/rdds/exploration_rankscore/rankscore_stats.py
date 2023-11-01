from h5py import File
import numpy as np
import seaborn as sns
import pandas as pd
from typing import List, Dict
from pprint import pprint

from . import WORKDIR
from .rankscore_performance import compute_optimal_performance_point
_DPI = 300

def histplot_smooth(serie: pd.Series, image_prefix: str):
    """
    Create a smoothened, continuous histplot
    :param serie:
    :param image_prefix: Prefix to image name
    :return:
    """
    fig: sns.FacetGrid = sns.displot(serie, kind='kde', bw_adjust=0.75)
    fig.set_xlabels(serie.name)
    fig.set_ylabels('KDE')
    fig.savefig(WORKDIR+f'/{image_prefix}{serie.name}_kernel_density_estimate.png', dpi=_DPI)


def cdfplot(serie:pd.Series, image_prefix: str):
    """
    Create a CDF plot
    :param serie:
    :return:
    """
    fig: sns.FacetGrid = sns.displot(serie, kind='ecdf')
    fig.set_xlabels(serie.name)
    fig.set_ylabels('eCDF')
    fig.savefig(WORKDIR+f'/{image_prefix}{serie.name}_empirical_culmulative_density_function.png', dpi=_DPI)


def rankscore_stats(file_path: str,
                    hd5_group_name: str ='structured_vcfs',
                    features: List[str] = None,
                    image_name_prefix: str = None,
                    k_fold_subset_size: int = None):
    """
    :param file_path:
    :param hd5_group_name: Group name in HD5 file that contains data to be analyzed
    :param features: List of features to be analyzed
    :param image_name_prefix: Prefix to add to saved image names (as image name differentiator).
    :return:
    """
    h5py_file: File = File(file_path, 'r')

    image_name_prefix = '' if image_name_prefix is None else image_name_prefix + '-'

    if features is None:
        features = ['RankScore_value',
                    'RankScoreNormalized_value',
                    'RankScoreMinMax_min',
                    'RankScoreMinMax_max']
    data: Dict[str, np.ndarray] = dict()
    for feature in features:
        try:
            feature_data: np.ndarray = h5py_file[hd5_group_name][feature][:, 0]
            data.update({feature: feature_data})
        except KeyError as error:
            print(f'Feature {feature} won\'t be analyzed: {error}')
    df: pd.DataFrame = pd.DataFrame(data=data)

    for column in df.columns:
        histplot_smooth(df[column], image_name_prefix)
        cdfplot(df[column], image_name_prefix)

    # TODO: Create proper split between train, test sets
    for rank_score_name in ['RankScore_value', 'RankScoreNormalized_value']:
        print(f'## Analyzing {rank_score_name} ##')
        rank_scores: np.ndarray = h5py_file[hd5_group_name][rank_score_name][:, 0]
        ground_truth_labels: np.ndarray = h5py_file[hd5_group_name]['label'][:, 0]
        metrics = compute_optimal_performance_point(rank_scores=rank_scores,
                                                    labels=ground_truth_labels,
                                                    save_plots=True,
                                                    k_fold_subset_size=k_fold_subset_size,
                                                    image_name_prefix=image_name_prefix+rank_score_name)
        pprint(metrics)
    print('Rankscore stats complete')
    # TODO: Store metrics as .json file
