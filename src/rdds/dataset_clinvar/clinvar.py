from urllib.request import urlretrieve
from os.path import join, basename
from . import WORKDIR
from typing import *
from rdds.lib.checksum import checksum


class Clinvar:

    def __init__(self,
                 vcf_file: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2023/clinvar_20230617.vcf.gz',
                 vcf_file_md5: str = 'da3660095d2061cf350dd670e5d68885',
                 gene_associations: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id',
                 gene_associations_md5: str = 'b2b695340baebd0c3ffae90dcff8266c',
                 disease_names: str = 'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/disease_names',
                 disease_names_md5: str = '9ba64343a03899de2e6cfb249caa95ad'):
        """
        Clinvar database adaptor.
        :param vcf_file: URL to VCF file containing clinvar variants
        :param vcf_file_md5: Checksum
        :param gene_associations: URL to file containing gene associations
        :param gene_associations_md5: Checksum
        :param disease_names: URL to file containing disease names
        :param disease_names_md5: Checksum
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
            if file_checksum != expected_checksum:
                raise ValueError(f'Failed md5 checksum: {storage_path}, got {file_checksum} expected {expected_checksum}')
