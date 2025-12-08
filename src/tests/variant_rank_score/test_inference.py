import subprocess as sp
import os
import shutil
import pandas as pd
import numpy as np
from numpy import isclose, sum
import pytest as pt
from datetime import datetime, timedelta

from rdds.lib.vcf import VCFReader, ParsableVariant
from rdds.variant_rank_score.model import VariantRankScoreModel

TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data.vcf'))
TEST_REFERENCE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data-predictions-ref.vcf'))


def parse_vrs_explanations(explanation: str) -> dict:
    """
    Parse MivmirExplanation field
    :param explanation: INFO key field contents for MivmirExplanation
    :return: Parsed dict, parts in explanation split into dict content
    """
    #
    explanation = explanation.lstrip('[')
    explanation = explanation.rstrip(']')
    explanation_dict = {}
    if len(explanation) > 0:  # Test for empty explanation field
        explanation = explanation.split(',')
        for entry in explanation:
            if len(entry) > 0:  # Test for last entry in explanation field [...,]
                key, value = entry.split('=')
                explanation_dict.update({
                    key: float(value)
                })
    return explanation_dict


class LayerOutputMismatch(ValueError): pass


def _recursively_compare_arrays(output: np.ndarray, output_ref: np.ndarray):
    """
    Helper function to compare nD arrays in a recursive fashion.
    Method is throwing an error in case arrays output != output_ref.
    """
    is_all_empty = output.size == 0 and output_ref.size == 0
    is_equal_size = output.size == output_ref.size
    is_high_rank = output.ndim > 1 and output_ref.ndim > 1
    try:
        arr = output[0]
        is_nested_array = isinstance(arr, np.ndarray)
    except IndexError:
        is_nested_array = False
    if is_all_empty:
        return  # Nothing to check since there's no data
    if not is_equal_size:
        raise ValueError(f'Different sizes for arrays {output.shape}, {output_ref.shape}. Expected identical')
    if is_high_rank or is_nested_array:
        outer_dim_size = output.shape[0]
        for idx in range(0, outer_dim_size):
            _recursively_compare_arrays(output=output[idx],
                                        output_ref=output_ref[idx])
    else:
        # Now expect a 1D array
        err = np.sum(output - output_ref)
        if not np.isclose(err, 0, atol=1E-5):
            raise LayerOutputMismatch(f'err={err}')

def test_inference_single_vs_batch(work_dir):
    """
    Test to see that magnitude of data/ composition does not
    affect individual variant scoring.

    This tests compare single execution scoring to batched scoring behavior.
    """

    # GIVEN some input data and model
    test_data_path = os.path.basename(TEST_DATA_PATH)
    test_data_path = os.path.join(work_dir, test_data_path)
    shutil.copyfile(TEST_DATA_PATH, test_data_path)
    vcf_reader = VCFReader(test_data_path)
    variants = list(vcf_reader)
    vcf_reader.close()
    parsed_variants = [ParsableVariant(variant=variant,
                                       vep_csq_description=vcf_reader.csq_description) for variant in variants]

    batch_predictions = dict()
    batch_sizes = [1, 2, 5, len(parsed_variants), len(parsed_variants) * 2]
    # WHEN running inference on a set of variants
    for batch_size in batch_sizes:
        model = VariantRankScoreModel()
        model.load_saved_model()
        scores = None
        for idx in range(0, len(parsed_variants), batch_size):
            batch_variants = parsed_variants[idx: idx + batch_size]
            score_batch = model.score_variant(variants=batch_variants)
            if scores is None:
                scores = score_batch
            else:
                scores = pd.concat((scores, score_batch), axis=0)
        batch_predictions.update({batch_size: scores})

    # THEN expect variant scoring to be identical regardless of batch composition
    for batch_size, df in batch_predictions.items():
        for batch_size_inner, df_inner in batch_predictions.items():
            predictions = df.pathogenicity_score.values
            predictions_inner = df_inner.pathogenicity_score.values
            err = sum(predictions - predictions_inner)
            assert isclose(err, 0, atol=1E-5), (predictions, predictions_inner, batch_size, batch_size_inner)


@pt.mark.parametrize('n_cores', [1, 2, 10, 20])
def test_inference(work_dir, n_cores):
    """
    Test for model inference.

    Compare to known behavior.
    """
    # GIVEN a VCF as input
    test_data_path = os.path.basename(TEST_DATA_PATH)
    test_data_path = os.path.join(work_dir, test_data_path)
    shutil.copyfile(TEST_DATA_PATH, test_data_path)
    # WHEN running inference
    t_start = datetime.now()
    sp.check_call(f'python3 -m rdds.variant_rank_score predict-on-vcf --cpu_cores {n_cores} {test_data_path}', shell=True, stderr=sp.STDOUT)
    t_stop = datetime.now()
    duration = t_stop - t_start
    # THEN expect inference to complete within reasonable time (1core 20s, 10 cores 13s, ...) locally.
    # On github workers, performance is bad so account for this.
    assert duration <= timedelta(seconds=60), f"Inference took too long: {duration}"
    model_output_file = test_data_path.replace('.vcf', '-predictions.vcf')
    # THEN expect that for every variant, the inference behavior is unchanged
    vcf_reader = VCFReader(model_output_file)
    vcf_reader_ref = VCFReader(TEST_REFERENCE_DATA_PATH)
    variants = list(vcf_reader)
    variants_ref = list(vcf_reader_ref)
    vcf_reader.close()
    vcf_reader_ref.close()
    for variant_ref in variants_ref:
        is_checked = False
        for variant in variants:
            if variant.ID == variant_ref.ID and \
            variant.CHROM == variant_ref.CHROM and \
            variant.POS == variant_ref.POS:
                # Test model inference
                # Model prediction comes with a precision of 5 decimal points (rest is just noise in comparison)
                assert isclose(variant.INFO['MivmirScore'],
                               variant_ref.INFO['MivmirScore'],
                               atol=1E-5), (variant.INFO['MivmirScore'],
                                            variant_ref.INFO['MivmirScore'],
                                            variant.ID, variant_ref.ID)
                # Test model explanations
                explanation = parse_vrs_explanations(variant.INFO['MivmirExplanation'])
                explanation_ref = parse_vrs_explanations(variant_ref.INFO['MivmirExplanation'])
                if len(explanation_ref) > 0:
                    # Model explanations comes with a precision of 2 decimal points
                    for key in explanation_ref.keys():
                        assert isclose(explanation_ref[key],
                                       explanation[key],
                                       atol=1E-2), (explanation, explanation_ref, variant.ID, variant_ref.ID, key)
                is_checked = True
        assert is_checked, f'Variant {variant_ref} is missing in predicted VCF'

@pt.mark.parametrize('ignore_clinvar_uncertain_conflicting_annotations', [True, False])
@pt.mark.parametrize('explain_variant_score_threshold', [0.9, 1.0])
@pt.mark.parametrize('shuffle_batch', [False, True])
def test_inference_batch_composition(ignore_clinvar_uncertain_conflicting_annotations,
                                     explain_variant_score_threshold,
                                     shuffle_batch):
    """
    Test for deterministic inference behavior across shuffled batches.
    """
    from rdds.variant_rank_score.model import VariantRankScoreModel
    from random import shuffle

    vcf_reader = VCFReader(TEST_DATA_PATH)
    variants = list(vcf_reader)

    n_batches = 5

    # GIVEN some (potentially shuffled) batches
    indexes = []
    for _ in range(0, n_batches):
        idx = list(range(0, vcf_reader.number_of_variants))
        if shuffle_batch:
            shuffle(idx)
        indexes.append(idx)

    # WHEN scoring the variants
    results = []
    for index in indexes:
        # Create a batch
        variants_subset = []
        for i in index:
            parsed_variant = ParsableVariant(variant=variants[i],
                                             vep_csq_description=vcf_reader.csq_description)
            variants_subset.append(parsed_variant)
        model = VariantRankScoreModel()
        model.load_saved_model()
        iter_prediction_df = model.score_variant(variants=variants_subset,
                                                 explain_variant_score_threshold=explain_variant_score_threshold,
                                                 ignore_clinvar_uncertain_conflicting_annotations=ignore_clinvar_uncertain_conflicting_annotations)
        # Append the IDs to the DF
        variant_ids = [variant.ID for variant in variants_subset]
        iter_prediction_df['ID'] = variant_ids
        iter_prediction_df.set_index('ID', inplace=True)
        results.append(iter_prediction_df)

    # THEN expect variant scoring to behave identically across all batches
    ref = results[0]
    for result in results:
        for column in ref.columns:
            d = ref[column] - result[column]
            d = d.dropna()
            err = np.sum(np.abs(d.values))
            assert np.isclose(err, 0.0, atol=1E-3), (column, d)
