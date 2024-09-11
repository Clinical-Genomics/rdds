from urllib.request import urlretrieve
from os.path import join, basename
from . import WORKDIR
from typing import *
from rdds.lib.checksum import checksum

# TODO: Lock down or version control gene_condition_source_id, disease_names files with checksums. Edited on a daily basis.

class Clinvar:

    def __init__(self,
                 vcf_file: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh37/archive_2.0/2023/clinvar_20230617.vcf.gz',
                 vcf_file_md5: str = '0f2762a2ef532e7db04ff4bab2ca49b9',
                 gene_associations: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id',
                 gene_associations_md5: str = None,
                 disease_names: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/disease_names',
                 disease_names_md5: str = None):
        """
        Clinvar database adaptor.
        :param vcf_file: URL to VCF file containing clinvar variants
        :param vcf_file_md5: Checksum
        :param gene_associations: URL to file containing gene associations
        :param gene_associations_md5: Checksum or None
        :param disease_names: URL to file containing disease names
        :param disease_names_md5: Checksum or None
        """
        self._vcf_file: str = vcf_file
        self._vcf_file_md5: str = vcf_file_md5
        self._gene_associations: str = gene_associations
        self._gene_associations_md5: str = gene_associations_md5
        self._disease_names: str = disease_names
        self._disease_names_md5: str = disease_names_md5

        self._download_files = [
            (self._vcf_file, self._vcf_file_md5),
            (self._gene_associations, self._gene_associations_md5),
            (self._disease_names, self._disease_names_md5)
        ]

    def download(self):
        for upstream_file_url, expected_checksum in self._download_files:
            storage_path = join(WORKDIR, basename(upstream_file_url))
            urlretrieve(upstream_file_url, storage_path)

            file_checksum: str = checksum(file_path=storage_path, algorithm='md5')

            if expected_checksum is None:
                print(f'{storage_path}, md5 {file_checksum}')
                continue

            if file_checksum != expected_checksum:
                raise ValueError(f'Failed md5 checksum: {storage_path}, got {file_checksum} expected {expected_checksum}')
