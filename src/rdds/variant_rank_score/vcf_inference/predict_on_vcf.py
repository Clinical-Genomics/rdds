import os
import pandas as pd
from cyvcf2 import Writer as VcfWriter
import gc

from rdds.lib.logging import get_logger
from .. import WORKDIR
from rdds.lib.vcf import VCFReader, ParsableVariant

from rdds.lib.vcf_inference import map_vcf

_LOGGER = get_logger('vrs_predict', 'info')


def _subprocess_predict_on_vcf_part(vcf_file_path: str,
                                    vrs_model_file_path: str,
                                    model_explainer_path: str,
                                    subprocess_work_dir: str,
                                    variant_index_start: int,
                                    variant_index_stop: int):
    from ..model import VariantRankScoreModel
    vcf_reader = VCFReader(vcf_file_path)
    vcf_reader.add_info_to_header({'ID': 'MivmirScore',
                                   'Description': 'Rank score from MIVMIR model (5 points precision)',
                                   'Type': 'Float',
                                   'Number': '1'})
    vcf_reader.add_info_to_header({'ID': 'MivmirExplanation',
                                   'Description': 'List of annotation impact scores on MivmirScore (2 points precision)',
                                   'Type': 'String',
                                   'Number': '.'})

    # Make a copy of the input VCF which is also the output file
    subprocess_output_file_name = os.path.join(subprocess_work_dir, f'{variant_index_start}.vcf')
    vcf_writer = VcfWriter(subprocess_output_file_name,
                           vcf_reader,  # Reuse original file header, with MivmirScore appended
                           mode='w')

    # Load and preprocess variants
    # Force load complete VCF into RAM as list of variants, drop out of scope variants
    variants = []
    for i, variant in enumerate(vcf_reader):
        # Get slice of all variants similar to [start:stop] but doesn't require all variants in RAM simultaneously
        if variant_index_stop > i >= variant_index_start:
            variants.append(variant)
    gc.collect()
    parsed_variants = []
    for i, variant in enumerate(variants):
        parsed_variants.append(ParsableVariant(variant=variant,  # Parsing of variants is really slow ...
                                               vep_csq_description=vcf_reader.csq_description))

    # Run model inference
    vrs_model = VariantRankScoreModel()
    vrs_model.load_saved_model(keras_model_path=vrs_model_file_path,
                               model_explainer_path=model_explainer_path)
    df: pd.DataFrame = vrs_model.score_variant(parsed_variants)
    for i, variant in enumerate(variants):
        df_i = df.iloc[i]
        variant.INFO['MivmirScore'] = f'{df_i.pathogenicity_score:.5F}'
        # Sort the explanations in decreasing importance (positive = more contributing to higher scoring result)
        explanations_sorted_in_decreasing_importance = df_i.sort_values(ascending=False)
        vrs_model_explanations = '['
        for key, contribution_score in list(explanations_sorted_in_decreasing_importance.items()):
            if key == 'pathogenicity_score':
                continue
            if not (contribution_score == contribution_score):  # NaN check
                continue
            vrs_model_explanations += f'{key}={contribution_score:.2F},'
        vrs_model_explanations += ']'
        variant.INFO['MivmirExplanation'] = vrs_model_explanations
        vcf_writer.write_record(variant)
    vcf_writer.close()
    vcf_reader.close()
    gc.collect()

def predict_on_vcf(vrs_model_file_path: str,
                   model_explainer_path: str,
                   vcf_file_path:str,
                   cpu_cores: int):
    """
    Run pre-trained model to annotate VCF with inferences.
    This method requires ~80GB RAM with default settings

    :param vrs_model_file_path: The path to the pretrained model
    :param model_explainer_path: The path to the saved model explainer
    :param vcf_file_path:
    :param cpu_cores:
    """
    fn_kwargs = {
        'vrs_model_file_path': vrs_model_file_path,
        'model_explainer_path': model_explainer_path,
    }
    map_vcf(vcf_file_path=vcf_file_path,
            fn=_subprocess_predict_on_vcf_part,
            fn_kwargs=fn_kwargs,
            cpu_cores=cpu_cores,
            logger=_LOGGER,
            workdir=WORKDIR)
