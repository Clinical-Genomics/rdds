from typing import Union, List
import os
from tempfile import mkdtemp
from rdds.lib import slurm
import math
from subprocess import check_call
from cyvcf2 import Writer as VcfWriter
import gc

from rdds.lib.list_dir import list_dir
from rdds.lib.logging import get_logger
from .. import WORKDIR
from rdds.lib.vcf import VCFReader, ParsableVariant

from rdds.lib.vcf_inference import map_vcf

_LOGGER = get_logger('gicam_infer', 'info')

def _infer_gicam_fn(vcf_file_path: str,
                    subprocess_work_dir: str,
                    variant_index_start: int,
                    variant_index_stop: int,
                    replace_overwrite_vrs_annotation = False):
    from rdds.gicam.model import Gicam
    gicam = Gicam.from_saved_model()

    vcf_reader = VCFReader(vcf_file_path)
    if not replace_overwrite_vrs_annotation:
        vcf_reader.add_info_to_header({'ID': 'GICAM',
                                       'Description': 'Rank score from GICAM model (joint MIVMIR and Genmod) (5 points precision)',
                                       'Type': 'Float',
                                       'Number': '1'})
    # TODO: Add GICAM version to header
    # Make a copy of the input VCF which is also the output file
    subprocess_output_file_name = os.path.join(subprocess_work_dir, f'{variant_index_start}.vcf')
    vcf_writer = VcfWriter(subprocess_output_file_name,
                           vcf_reader,  # Reuse original file header, with VrsModelPrediction appended
                           mode='w')
    # Load and preprocess variants
    # Force load complete VCF into RAM as list of variants, drop out of scope variants
    variants = []
    for i, variant in enumerate(vcf_reader):
        # Get slice of all variants similar to [start:stop] but doesn't require all variants in RAM simultaneously
        if variant_index_stop > i >= variant_index_start:
            variants.append(variant)
    gc.collect()
    scores = gicam.score_variants(variants=variants)
    for i, (variant, score) in enumerate(zip(variants, scores)):
        if replace_overwrite_vrs_annotation:
            variant.INFO['VrsModelPrediction'] = f'{score:.5f}'
        else:
            variant.INFO['GICAM'] = f'{score:.5f}'
        vcf_writer.write_record(variant)
    vcf_writer.close()
    vcf_reader.close()
    gc.collect()


def infer_vcf(vcf_file_path: str,
              cpu_cores: int,
              replace_overwrite_vrs_annotation = False) -> str:
    """
    Execute GICAM inference on VCF in multiproc fashion.
    :param vcf_file_path: Path to VCF to process
    :param cpu_cores: Amount of CPU cores to allocate
    :param replace_overwrite_vrs_annotation: Overwrite VRS model inference annotation
    :return path to processed VCF
    """
    kwargs = {
        'replace_overwrite_vrs_annotation': replace_overwrite_vrs_annotation
    }
    return map_vcf(vcf_file_path=vcf_file_path,
                   fn=_infer_gicam_fn,
                   fn_kwargs=kwargs,
                   cpu_cores=cpu_cores,
                   logger=_LOGGER,
                   workdir=WORKDIR)