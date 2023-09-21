from h5py import File, Group
import pandas as pd
from typing import Tuple, Any

from rdds.exploration_rankscore.rankscore_low_bound_cap_check import is_below_normalisation_low_bound
from rdds.exploration_rankscore.assemble_unique_variant_id import dataframe_assemble_unique_variant_id

"""
Comparison methods for evaluation of normalized rank score.

Part of testing schema related to introducing 'RankScoreNormalized'
in Genmod.
"""

_RANKSCORE_MAX_DIFF = float(1E-6)


def test_rankscore_normalization(dataset: Group):
    """
    Test RankScore and RankScoreNormalized integrity in dataset.
    * RankScore(s) are real valued numbers (not NaN)
    * RankScoreNormalized in range (0, 1)
    * RankScore is in range of MinMax normalization bounds
    * Performing MinMax normalization of RankScore yields RankScoreNormalized
    :param dataset: The dataset to test
    :return:
    """
    print('Starting rankscore normalization test')

    def test_realvalued(id: str, v: float):
        if not v == v:
            raise ValueError(f'NaN value found variant_id={id} value={v}')

    df = pd.DataFrame(data={'rank_score': dataset['RankScore_value'][:, 0],
                            'rank_score_normalized': dataset['RankScoreNormalized_value'][:, 0],
                            'rank_score_min': dataset['RankScoreMinMax_min'][:, 0],
                            'rank_score_max': dataset['RankScoreMinMax_max'][:, 0],
                            'variant_ids': dataset['variant_ids'][:, 0]})
    if not len(df) > 0:
        raise ValueError('No variants in dataset')

    for row in df.itertuples():
        # Perform type/NaN check
        test_realvalued(row.variant_ids, row.rank_score)
        test_realvalued(row.variant_ids, row.rank_score_normalized)
        test_realvalued(row.variant_ids, row.rank_score_min)
        test_realvalued(row.variant_ids, row.rank_score_max)

        # Check RankScoreNormalized bounds
        if not 0.0 <= row.rank_score_normalized <= 1.0:
            raise ValueError(f'Normalized value outside expected bounds (0, 1) {row}')

        # Check normalization bounds
        if not row.rank_score_min <= row.rank_score_max:
            raise ValueError(f'Bad normalization bounds {row}')

        # Check RankScore -> RankScoreNormalized integrity by means of MinMax normalization
        rankscore_normalized_computed: float = (row.rank_score - row.rank_score_min) / (row.rank_score_max - row.rank_score_min)
        delta: float = abs(row.rank_score_normalized - rankscore_normalized_computed)
        if delta >= _RANKSCORE_MAX_DIFF:
            raise ValueError(f'Mismatch between rankscore normalized and recomputed\
             rankscore using min-max bounds {row} {rankscore_normalized_computed}')

    print(f'Completed rankscore normalization test, checked {len(df)} variants')


def test_compare_rankscore(dataset_ref: Group,
                           dataset: Group):
    """
    Compare RankScore in dataset to ranked variants in dataset_ref,
    to make sure they're ranked identically (i.e. the RankScore value
    is identical).

    This method assumes that the datasets contain an identical set of
    variants.

    :param dataset_ref: Reference dataset
    :param dataset: The dataset to test
    :return:
    """
    print('Starting rankscore data set comparison test')
    df_test: pd.DataFrame = pd.DataFrame(data={'rank_score': dataset['RankScore_value'][:, 0],
                                               'alt': dataset['alt'][:, 0],
                                               'ref': dataset['ref'][:, 0],
                                               'pos': dataset['pos'][:, 0],
                                               'normalization_min_bound': dataset['RankScoreMinMax_min'][:, 0]},
                                         index=dataset['variant_ids'][:, 0])
    df_test = dataframe_assemble_unique_variant_id(df=df_test)
    df_test.sort_index(inplace=True)
    if len(df_test) == 0:
        raise ValueError('Expected data in test set')
    df_ref: pd.DataFrame = pd.DataFrame(data={'rank_score_ref': dataset_ref['RankScore_value'][:, 0],
                                              'alt': dataset_ref['alt'][:, 0],
                                              'ref': dataset_ref['ref'][:, 0],
                                              'pos': dataset_ref['pos'][:, 0],
                                              }, index=dataset_ref['variant_ids'][:, 0])
    df_ref = dataframe_assemble_unique_variant_id(df=df_ref)
    df_ref.sort_index(inplace=True)
    if len(df_ref) == 0:
        raise ValueError('Expected data in ref set')
    if not len(df_ref) == len(df_test):
        raise ValueError(f'Mismatch in amount of variants in datasets, REF={len(df_ref)} TEST={len(df_test)}')
    df: pd.DataFrame = pd.concat((df_test, df_ref), axis=1)  # Concatenate DFs based on index (variant_id)

    def check_rank_score(row: Tuple[Any, ...]) -> None:
        """
        Checks RankScore so that they're identical in reference and test sets.
        :param row:
        :return:
        """
        if not row.rank_score_ref == row.rank_score_ref:  # ref set value is NaN (variant missing from ref set)
            return
        if abs(row.rank_score_ref - row.rank_score) >= _RANKSCORE_MAX_DIFF:
            if is_below_normalisation_low_bound(rank_score=row.rank_score_ref,
                                                rank_score_normalized=row.rank_score,
                                                rank_score_normalization_low_bound=row.normalization_min_bound):
                return  # Since the rank score is capped to min of rank score normalized bound by design, it's OK
            raise ValueError(f'Mismatch rank score {row}')

    # For every variant in dataset, make sure the rank score is identical to test set
    for row in df.itertuples():
        check_rank_score(row)

    print(f'Completed rankscore data set comparison test, checked {len(df)} variants')


def run_rankscore_normalization_tests(file_path_ref: str,
                                      file_path: str):
    """
    Evaluate contents of hd5 files with respect to RankScore and RankScoreNormalized fields.

    :param file_path_ref: Reference dataset
    :param file_path: Dataset under test
    :return:
    """
    hd5_file = File(file_path, 'r')
    dataset = hd5_file['structured_vcfs']

    test_rankscore_normalization(dataset)
    if file_path_ref:
        hd5_file_ref = File(file_path_ref, 'r')
        dataset_ref = hd5_file_ref['structured_vcfs']
        test_compare_rankscore(dataset_ref=dataset_ref,
                               dataset=dataset)
        hd5_file_ref.close()

    hd5_file.close()
