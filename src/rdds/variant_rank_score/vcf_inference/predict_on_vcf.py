import math
import os

import pandas as pd
import progressbar
from cyvcf2 import Writer as VcfWriter
from tempfile import mkdtemp
from subprocess import check_call
import gc
from multiprocessing import SimpleQueue

from rdds.lib.list_dir import list_dir
from rdds.lib.logging import get_logger
from rdds.lib.process_pool import ProcessPool
from .. import WORKDIR
from rdds.lib.vcf import VCFReader, ParsableVariant
from rdds.lib import slurm

_LOGGER = get_logger('vrs_predict', 'info')


def _subprocess_predict_on_vcf_part(vcf_file_path: str,
                                    vrs_model_file_path: str,
                                    model_explainer_path: str,
                                    subprocess_work_dir: str,
                                    variant_index_start: int,
                                    variant_index_stop: int):
    from ..model import VariantRankScoreModel
    vcf_reader = VCFReader(vcf_file_path)
    vcf_reader.add_info_to_header({'ID': 'VrsModelPrediction',
                                   'Description': 'Rank score from VRS model (5 points precision)',
                                   'Type': 'Float',
                                   'Number': '1'})
    vcf_reader.add_info_to_header({'ID': 'VrsModelExplanation',
                                   'Description': 'List of annotation impact scores on VrsModelPrediction (2 points precision)',
                                   'Type': 'String',
                                   'Number': '.'})

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
        variant.INFO['VrsModelPrediction'] = f'{df_i.pathogenicity_score:.5F}'
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
        variant.INFO['VrsModelExplanation'] = vrs_model_explanations
        vcf_writer.write_record(variant)
    vcf_writer.close()
    vcf_reader.close()
    gc.collect()


def predict_on_vcf(vrs_model_file_path: str,
                   model_explainer_path: str,
                   vcf_file_path: str,
                   cpu_cores: int,
                   max_batch_size_per_worker: int = int(10E4)
                   ):
    """
    Run pre-trained model to annotate VCF with inferences.
    This method requires ~80GB RAM with default settings

    :param vrs_model_file_path: The path to the pretrained model
    :param model_explainer_path: The path to the saved model explainer
    :param vcf_file_path: The path to the VCF file
    :param cpu_cores: Maximum amount of CPU cores to allocate to job
    :param max_batch_size_per_worker: Amount of variants per worker subprocess task (partition due to limited RAM)
    :return: Prints annotated file path on stdout
    """

    # TODO: cyvcf2 currently does not support writing string type format fields with number>1.

    if not '.vcf' in vcf_file_path:
        raise ValueError(f'Expected a VCF but got {vcf_file_path}')
    new_file_name = os.path.basename(vcf_file_path).replace('.vcf', '-predictions.vcf')
    annotated_vcf_file_path = os.path.join(os.path.dirname(vcf_file_path), new_file_name)
    if annotated_vcf_file_path == vcf_file_path:
        raise ValueError(f'Won\'t overwrite existing VCF file')

    subprocess_work_dir = mkdtemp(prefix='vcf-predict-', dir=WORKDIR)

    # Count number of variants in input VCF
    vcf_reader = VCFReader(vcf_file_path)
    n_variants = vcf_reader.number_of_variants
    vcf_reader.close()
    del vcf_reader

    n_workers = min(cpu_cores, slurm.cpu_count())
    batch_size = int(math.ceil((n_variants / float(n_workers))))
    batch_size = max_batch_size_per_worker if batch_size > max_batch_size_per_worker else batch_size
    subprocess_args = []
    worker_names = []
    for variant_idx in range(0, n_variants, batch_size):
        subprocess_args.append((vcf_file_path,
                                vrs_model_file_path,
                                model_explainer_path,
                                subprocess_work_dir,
                                variant_idx,
                                variant_idx + batch_size))
        worker_names.append(f"{variant_idx}-{variant_idx+batch_size}")
    _LOGGER.info(f'Worker load: {n_workers} workers \
with {batch_size:.0E} variants per worker, \
totalling {len(subprocess_args)} worker tasks')
    pool = ProcessPool(function=_subprocess_predict_on_vcf_part,
                       args=subprocess_args,
                       process_names=worker_names,
                       workers=n_workers)
    task_queue = pool.run_async()
    pbar = progressbar.ProgressBar(max_value=len(subprocess_args),
                                   prefix='Total processing progress ')
    completed_tasks = []
    while True:
        task = task_queue.get(timeout=60*60*5)  # Raises Empty exception if no data after timeout
        if task.process.exitcode != 0:
            raise ValueError(f'Task failed: {task}')
            break
        completed_tasks.append(task)
        if len(completed_tasks) == len(subprocess_args):
            break
        pbar.update(len(completed_tasks))
    pbar.finish()
    pool.close()

    # Merge VCFs to produce final output file
    # TODO: Check that all files are non-empty and submitted jobs
    vcf_parts = list(list_dir(directory_path=subprocess_work_dir))
    vcf_parts = sorted(vcf_parts, key=lambda path: int(os.path.basename(path.rstrip('.vcf'))))
    cmdline = "bcftools concat --no-version "
    for vcf_part in vcf_parts:
        cmdline += "%s " % vcf_part
    cmdline += " > %s" % annotated_vcf_file_path
    check_call(cmdline, shell=True)
    [os.remove(vcf_part) for vcf_part in vcf_parts]
    os.rmdir(subprocess_work_dir)
    _LOGGER.info(f'Completed. Output file: {annotated_vcf_file_path}')

