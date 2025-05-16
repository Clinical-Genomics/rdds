import subprocess as sp
import os
import shutil
import pandas as pd
import numpy as np
from numpy import isclose, sum
import pytest as pt

from rdds.lib.vcf import VCFReader, ParsableVariant
from rdds.variant_rank_score.model import VariantRankScoreModel

TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data.vcf'))
TEST_REFERENCE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data-predictions-ref.vcf'))


def parse_vrs_explanations(explanation: str) -> dict:
    """
    Parse VrsModelExplanation field
    :param explanation: INFO key field contents for VrsModelExplanation
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


def test_inference_single_vs_batch(work_dir):
    """
    Test to see that magnitude of data/ composition does not
    affect individual variant scoring.
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
        # TODO: Assemble batch_variants randomly from parsed_variants
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
    sp.check_call(f'python3 -m rdds.variant_rank_score predict-on-vcf --cpu_cores {n_cores} {test_data_path}', shell=True, stderr=sp.STDOUT)
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
                assert isclose(variant.INFO['VrsModelPrediction'],
                               variant_ref.INFO['VrsModelPrediction'],
                               atol=1E-5), (variant.INFO['VrsModelPrediction'],
                                            variant_ref.INFO['VrsModelPrediction'],
                                            variant.ID, variant_ref.ID)
                # Test model explanations
                explanation = parse_vrs_explanations(variant.INFO['VrsModelExplanation'])
                explanation_ref = parse_vrs_explanations(variant_ref.INFO['VrsModelExplanation'])
                if len(explanation_ref) > 0:
                    # Model explanations comes with a precision of 2 decimal points
                    for key in explanation_ref.keys():
                        assert isclose(explanation_ref[key],
                                       explanation[key],
                                       atol=1E-2), (explanation, explanation_ref, variant.ID, variant_ref.ID)
                is_checked = True
        assert is_checked, f'Variant {variant_ref} is missing in predicted VCF'


@pt.mark.parametrize('ignore_clinvar_uncertain_conflicting_annotations', [True, False])
@pt.mark.parametrize('explain_variant_score_threshold', [0.9, 1.0])
def test_inference_reproducibility(ignore_clinvar_uncertain_conflicting_annotations,
                                   explain_variant_score_threshold):
    """
    Run inference 10 times per variant, to check inference reproducibility.

    Disable model explanation step since it's too computationally expensive.
    """
    from rdds.variant_rank_score.model import VariantRankScoreModel
    import gc

    # GIVEN a model
    # WHEN computing inferences
    vcf_reader = VCFReader(TEST_DATA_PATH)
    for variant in vcf_reader:
        parsed_variant = ParsableVariant(variant)
        reference_score: float = None
        # THEN expect variant scoring to behave identically if recomputed
        for _ in range(0, 5):
            model = VariantRankScoreModel()
            model.load_saved_model()
            iter_prediction_df = model.score_variant(variants=[parsed_variant],
                                                     explain_variant_score_threshold=explain_variant_score_threshold,
                                                     ignore_clinvar_uncertain_conflicting_annotations=ignore_clinvar_uncertain_conflicting_annotations
                                                     )
            pathogenicity_score = iter_prediction_df.pathogenicity_score.values[0]
            if reference_score is None:
                reference_score = pathogenicity_score
            assert isclose(reference_score, pathogenicity_score, atol=1E-6)
            del model
            gc.collect()
    vcf_reader.close()