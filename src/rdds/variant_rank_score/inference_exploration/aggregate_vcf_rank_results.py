import os.path
from typing import List, Dict, Any
from os.path import join
from json import loads
import matplotlib.pyplot as plt
import pandas as pd

FIGSIZE = (30, 20)

def aggregate_vcf_rank_results(view_rank_result_output_dir: str,
                               case_names: List[str]):
    """
    Comptute aggreate performance results across a set of cases.

    - Compare genmod and VRS model ranks
    """
    # Load json file containing serialized pandas dataframes back to DataFrame format
    case_metas: Dict[str, Dict[str, Any]] = {}
    for case_name in case_names:
        meta_file_path = join(view_rank_result_output_dir, case_name, 'performance.json')
        print(f'Loading {meta_file_path}')
        with open(meta_file_path, 'r') as file:
            data_dict: Dict[str, Any] = loads(file.read())
            print(data_dict)
            for sub_dict_key, sub_dict in data_dict.items():
                df = pd.DataFrame.from_dict(sub_dict)
                data_dict[sub_dict_key] = df
            case_metas.update({case_name: data_dict})

    # Assemble a plot case name as x, rank score as y for genmod and vrs model
    dict_rank_comparison = {}
    for case_name, case_meta in case_metas.items():
        dict_rank_comparison.update({
            case_name: {
                'vrs_rank': case_meta['vrs_rank'].index.values.astype(int),
                'vrs_rank_frqfilt': case_meta['vrs_rank_frqfilt'].index.values.astype(int),
                'genmod_rank': case_meta['genmod_rank'].index.values.astype(int)
            }
        })
    # df[index=case_names, columns=[rank score of models]
    df_rank_comparison = pd.DataFrame.from_dict(dict_rank_comparison, orient='index')
    df_rank_comparison['case_name'] = df_rank_comparison.index
    df_rank_comparison.reset_index(inplace=True)
    # TODO: Refactor input file format to remove below hacky flattening code
    df_rank_comparison = df_rank_comparison.explode('vrs_rank')
    df_rank_comparison = df_rank_comparison.explode('vrs_rank_frqfilt')
    df_rank_comparison = df_rank_comparison.explode('genmod_rank')

    wdir = os.path.join(os.path.dirname(view_rank_result_output_dir), 'aggregate-statistics')
    os.makedirs(wdir, exist_ok=True)

    points_vrs = 0
    points_genmod = 0
    ties = 0
    for index, row in df_rank_comparison.iterrows():
        if row.genmod_rank > row.vrs_rank_frqfilt:
            points_vrs += 1
        elif row.genmod_rank < row.vrs_rank_frqfilt:
            points_genmod += 1
        elif row.genmod_rank == row.vrs_rank_frqfilt:
            ties += 1
        else:
            raise ValueError(row)


    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.scatter(x=df_rank_comparison.case_name,
                y=df_rank_comparison['genmod_rank'],
               marker='o',
               alpha=0.75)
    # FIXME: The vrs nonfilt rank is messing with the dynrange. Plot separately
    #ax.scatter(x=df_rank_comparison.case_name,
    #            y=df_rank_comparison['vrs_rank'])
    ax.scatter(x=df_rank_comparison.case_name,
                y=df_rank_comparison['vrs_rank_frqfilt'],
               marker='D',
               alpha=0.75)
    ax.grid(True)
    ax.legend(['genmod', 'vrs-frqfilt'])
    fig.tight_layout()
    fig.suptitle(f'Pathogenic Variant Rank Per Model\nvrs {points_vrs}, genmod {points_genmod}, ties {ties}')
    fig_path = os.path.join(wdir, 'rank.png')
    fig.savefig(fig_path)
    print(f'Saving figure {fig_path}. Completed aggregate analysis.')

    return
