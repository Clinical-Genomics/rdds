import os.path
import subprocess
from time import time
from typing import List
from os import cpu_count

from .. import WORKDIR


class VCFDataSet:

    """
    Helper to deal with VCF datasets.
    """

    @staticmethod
    def concat_datasets(vcf_dataset_paths: List[str]) -> str:
        """
        Concatenates vcf files into a single VCF record.
        param: vcf_datasets_path: List of paths to VCF to be concatenated (bgzipped)
        """
        for path in vcf_dataset_paths:
            if not os.path.exists(path):
                raise ValueError(f'{path} does not exist')
            if not '.gz' in path:
                raise ValueError(f'Expected a bgzipped file, got {path}')
        paths: str = ''
        for path in vcf_dataset_paths:
            paths += f' {path}'
        output_file: str = os.path.join(WORKDIR, f'concat-dataset-{int(time())}.vcf')
        print(f'Concatenating {vcf_dataset_paths} > {output_file}.gz')
        subprocess.check_call(f'bcftools concat --allow-overlaps {paths} -o {output_file}',
                              stderr=subprocess.STDOUT,
                              shell=True)
        subprocess.call(f'bgzip -f --threads {cpu_count()} {output_file}',
                        stderr=subprocess.STDOUT,
                        shell=True)
        subprocess.call(f'tabix -f -p vcf {output_file}.gz',
                        stderr=subprocess.STDOUT,
                        shell=True)
        return output_file+'.gz'
