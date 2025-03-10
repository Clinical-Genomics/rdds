import subprocess as sp
import os
import shutil
from numpy import isclose

from rdds.lib.vcf import VCFReader, ParsableVariant

TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data.vcf'))

EXPECTED_VALUES ={'1-1336445': {'label': 'Likely_benign', 'vrs_model_prediction': 0.0036100000143051147, 'vrs_model_explanation': {}},
                  '1-976215': {'label': 'Pathogenic', 'vrs_model_prediction': 0.2750200033187866, 'vrs_model_explanation': {}},
                  '2-47403394': {'label': 'Likely_pathogenic', 'vrs_model_prediction': 0.9149900078773499, 'vrs_model_explanation': {'most_severe_consequence': 0.33, 'ModelScore_value': 0.28, 'CADD': 0.15, 'CSQ_CLINVAR_CLNSIG': 0.04, 'CSQ_MES-SWA_donor_alt': 0.04, 'SPIDEX': 0.02, 'SWEGENAF': 0.01, 'Frq': 0.01, 'CSQ_SIFT': 0.01, 'CSQ_MaxEntScan_diff': 0.01, 'CSQ_MES-SWA_acceptor_alt': 0.01, 'CSQ_MES-SWA_donor_diff': 0.0, 'CSQ_SpliceAI_pred_DS_AG': 0.0, 'GNOMADAF_popmax': 0.0, 'CSQ_CLINVAR_CLNREVSTAT': 0.0, 'CSQ_phyloP100way_vertebrate': 0.0, 'CSQ_phastCons100way_vertebrate': 0.0, 'CSQ_GERP++_RS': 0.0, 'CSQ_REVEL_score': 0.0, 'CSQ_PolyPhen': 0.0, 'CSQ_SpliceAI_pred_DS_DG': 0.0, 'CSQ_SpliceAI_pred_DS_AL': 0.0, 'CSQ_SpliceAI_pred_DS_DL': 0.0, 'CSQ_LoFtool': -0.0, 'CSQ_MaxEntScan_alt': -0.0}},
                  '3-33114704': {'label': 'Benign', 'vrs_model_prediction': 0.0, 'vrs_model_explanation': {}},
                  '4-1803751': {'label': 'Benign', 'vrs_model_prediction': 0.8599100112915039, 'vrs_model_explanation': {}},
                  '5-45645297': {'label': 'Pathogenic', 'vrs_model_prediction': 0.8604099750518799, 'vrs_model_explanation': {}},
                  '5-45645306': {'label': 'Pathogenic', 'vrs_model_prediction': 0.9663599729537964, 'vrs_model_explanation': {'ModelScore_value': 0.39, 'CSQ_REVEL_score': 0.3, 'CADD': 0.12, 'most_severe_consequence': 0.1, 'CSQ_phyloP100way_vertebrate': 0.07, 'CSQ_CLINVAR_CLNSIG': 0.06, 'Frq': 0.01, 'CSQ_LoFtool': 0.0, 'CSQ_SpliceAI_pred_DS_DL': 0.0, 'CSQ_MaxEntScan_alt': 0.0, 'CSQ_SpliceAI_pred_DS_AG': 0.0, 'SPIDEX': 0.0, 'GNOMADAF_popmax': 0.0, 'SWEGENAF': 0.0, 'CSQ_SIFT': 0.0, 'CSQ_CLINVAR_CLNREVSTAT': 0.0, 'CSQ_MES-SWA_acceptor_alt': 0.0, 'CSQ_MES-SWA_donor_alt': 0.0, 'CSQ_MaxEntScan_diff': 0.0, 'CSQ_SpliceAI_pred_DS_DG': 0.0, 'CSQ_SpliceAI_pred_DS_AL': 0.0, 'CSQ_MES-SWA_donor_diff': 0.0, 'CSQ_PolyPhen': -0.02, 'CSQ_phastCons100way_vertebrate': -0.03, 'CSQ_GERP++_RS': -0.04}},
                  '6-7580872': {'label': 'Pathogenic', 'vrs_model_prediction': 0.006829999852925539, 'vrs_model_explanation': {}},
                  '6-7580957': {'label': 'Benign/Likely_benign', 'vrs_model_prediction': 0.0010100000072270632, 'vrs_model_explanation': {}},
                  '7-50458565': {'label': 'Benign', 'vrs_model_prediction': 0.0036299999337643385, 'vrs_model_explanation': {}},
                  '7-50463289': {'label': 'Pathogenic', 'vrs_model_prediction': 0.0009800000116229057, 'vrs_model_explanation': {}},
                  '9-13193319': {'label': 'Benign', 'vrs_model_prediction': 0.004970000125467777, 'vrs_model_explanation': {}},
                  '19-13207585': {'label': 'Likely_pathogenic', 'vrs_model_prediction': 0.0, 'vrs_model_explanation': {}}}


def test_inference(work_dir):
    """
    Test for model inference.

    Compare to known behavior.
    """
    # GIVEN a VCF as input
    test_data_path = os.path.basename(TEST_DATA_PATH)
    test_data_path = os.path.join(work_dir, test_data_path)
    shutil.copyfile(TEST_DATA_PATH, test_data_path)
    # WHEN running inference
    sp.check_call(f'python3 -m rdds.variant_rank_score predict-on-vcf {test_data_path}', shell=True)
    model_output_file = test_data_path.replace('.vcf', '-predictions.vcf')
    # THEN expect that for every variant, the inference behavior is unchanged
    vcf_reader = VCFReader(model_output_file)
    scored_variants = dict()
    for variant in vcf_reader:
        # Parse VrsModelExplanation field
        explanation: str = variant.INFO['VrsModelExplanation']
        print(f'{variant.ID} EXP {explanation}')
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

        scored_variants.update({
            f'{variant.CHROM}-{variant.POS}': {
                'label': variant.INFO['CLINVAR_GROUND_TRUTH'],
                'vrs_model_prediction': variant.INFO['VrsModelPrediction'],
                'vrs_model_explanation': explanation_dict
            }
        })
    #print(scored_variants)  To view updated results, to make changes to EXPECTED_VALUES

    vcf_reader.close()
    for key in EXPECTED_VALUES.keys():
        assert EXPECTED_VALUES[key]['label'] == scored_variants[key]['label'], f'{scored_variants[key]}'
        # Model prediction comes with a precision of 5 decimal points (rest is just noise in comparison)
        assert isclose(EXPECTED_VALUES[key]['vrs_model_prediction'],
                       scored_variants[key]['vrs_model_prediction'],
                       atol=1E-5)
        for key_explain in EXPECTED_VALUES[key]['vrs_model_explanation'].keys():
            if len(EXPECTED_VALUES[key]['vrs_model_explanation']) == 0 and \
                len(scored_variants[key]['vrs_model_explanation']) == 0:
                # If model explanation is empty dict
                continue
            # Model explanations comes with a precision of 2 decimal points
            assert isclose(EXPECTED_VALUES[key]['vrs_model_explanation'][key_explain],
                           scored_variants[key]['vrs_model_explanation'][key_explain],
                           atol=1E-2), (key, key_explain)

    assert False, 'STOP'

def test_inference_reproducibility():
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
        for _ in range(0, 10):
            model = VariantRankScoreModel()
            model.load_saved_model()
            iter_prediction_df = model.score_variant(variants=[parsed_variant],
                                                     explain_variant_score_threshold=100.0)
            pathogenicity_score = iter_prediction_df.pathogenicity_score.values[0]
            if reference_score is None:
                reference_score = pathogenicity_score
            assert isclose(reference_score, pathogenicity_score, atol=1E-6)
            del model
            gc.collect()
    vcf_reader.close()