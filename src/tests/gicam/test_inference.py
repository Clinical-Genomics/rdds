import pytest as pt
import os
import subprocess as sp
import shutil
import numpy as np

from rdds.lib.vcf import VCFReader, Variant
from rdds.gicam.model import Gicam
from rdds.gicam.vcf_inference.infer_vcf import infer_vcf

TEST_DATA_PATH="/rdds/src/tests/gicam/inference_test_data.vcf"

@pt.mark.parametrize('overwrite_vrs_annotation', [False, True])
def test_vcf_inference(work_dir, overwrite_vrs_annotation):
    test_data_path = os.path.basename(TEST_DATA_PATH)
    test_data_path = os.path.join(work_dir, test_data_path)
    shutil.copyfile(TEST_DATA_PATH, test_data_path)
    output_file = test_data_path.replace('.vcf', '-predictions.vcf')

    infer_vcf(vcf_file_path=test_data_path, cpu_cores=1, replace_overwrite_vrs_annotation=overwrite_vrs_annotation)
    reader = VCFReader(output_file, 'r')
    target_annotation = 'GicamScore'
    if overwrite_vrs_annotation:
        target_annotation = 'MivmirScore'
    assert target_annotation in reader.info_fields
    for variant in list(reader):
        assert isinstance(variant.INFO[target_annotation], float)
        assert 0 <= variant.INFO[target_annotation] <= 1
    reader.close()

@pt.mark.parametrize('cpu_cores', [1, 2, 10, 20])
def test_vcf_inference_cli(work_dir, cpu_cores):
    """
    Test for GICAM VCF inference.
    """
    test_data_path = os.path.basename(TEST_DATA_PATH)
    test_data_path = os.path.join(work_dir, test_data_path)
    shutil.copyfile(TEST_DATA_PATH, test_data_path)
    output_file = test_data_path.replace('.vcf', '-predictions.vcf')

    sp.check_call(f'python3 -m rdds.gicam infer-vcf --cpu_cores {cpu_cores} {test_data_path}',
                  shell=True, stderr=sp.STDOUT)

    reader = VCFReader(output_file, 'r')
    assert 'GicamScore' in reader.info_fields
    for variant in list(reader):
        assert isinstance(variant.INFO['GicamScore'], float)
        assert 0 <= variant.INFO['GicamScore'] <= 1
    reader.close()

def test_score_variant():
    gicam = Gicam.from_saved_model()
    reader = VCFReader(TEST_DATA_PATH, 'r')
    variants = list(reader)
    reader.close()
    scores = gicam.score_variants(variants=variants)
    assert not np.any(np.logical_not(np.isclose(scores, 0.5, atol=0.5))), scores
