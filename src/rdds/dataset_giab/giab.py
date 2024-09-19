from urllib.request import urlretrieve
from os.path import join, basename
from . import WORKDIR
from typing import *
from rdds.lib.checksum import checksum

class Giab:

    def __init__(self,
                 vcf_file: str = 'https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh37/SupplementaryFiles/HG002_GRCh37_1_22_v4.2.1_all.vcf.gz',
                 vcf_file_md5: str = '7c37e16504686b828c3b904da18b4295',
                 vcf_index_file: str = 'https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh37/SupplementaryFiles/HG002_GRCh37_1_22_v4.2.1_all.vcf.gz.tbi',
                 vcf_index_file_md5: str = '8260628c9d800391e28e4bc4f507d53e'):
        """
        Genome In a Bottle database adaptor.
        :param vcf_file: URL to VCF file containing GIAB/AshkenazimTrio/Son called variants.
        :param vcf_file_md5: Checksum
        """
        self._vcf_file: str = vcf_file
        self._vcf_file_md5: str = vcf_file_md5
        self._vcf_index_file: str = vcf_index_file
        self._vcf_index_file_md5: str = vcf_index_file_md5

        self._download_files = [
            (self._vcf_file, self._vcf_file_md5),
            (self._vcf_index_file, self._vcf_index_file_md5),
        ]

    def download(self):
        for upstream_file_url, expected_checksum in self._download_files:
            storage_path = join(WORKDIR, basename(upstream_file_url))
            urlretrieve(upstream_file_url, storage_path)

            file_checksum: str = checksum(file_path=storage_path, algorithm='md5')

            if file_checksum != expected_checksum:
                raise ValueError(f'Failed md5 checksum: {storage_path}, got {file_checksum} expected {expected_checksum}')
