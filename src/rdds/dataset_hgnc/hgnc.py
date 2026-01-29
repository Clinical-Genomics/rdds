from urllib.request import urlretrieve
from os.path import join, basename
from os import listdir, remove

from . import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.checksum import checksum

# https://www.genenames.org/download/archive/
_HGNC_URL = 'https://storage.googleapis.com/public-download-files/hgnc/archive/archive/quarterly/tsv'

_LOGGER = get_logger('dataset-hgnc', 'info')
class HGNC:

    def __init__(self):
        # FIXME: Add GRCh37 as well, this file is just 38
        self._hgnc_complete_set = _HGNC_URL + '/hgnc_complete_set_2026-01-06.txt'  # GRCh38
        self._hgnc_complete_set_sha256 = '744b245d30ae95cb91aae1039f9cb62b32246261565fe0039199aac5f230ff6e'
        self._files_checksums = [
            (self._hgnc_complete_set, self._hgnc_complete_set_sha256)
        ]

    def download(self):
        [remove(join(WORKDIR, file_path)) for file_path in listdir(WORKDIR)]
        for url, expected_checksum in self._files_checksums:
            storage_path = join(WORKDIR, basename(url))
            _LOGGER.info(f"Pulling {url} -> {storage_path}")
            urlretrieve(url, storage_path)
            file_checksum: str = checksum(file_path=storage_path, algorithm='sha256')
            if file_checksum != expected_checksum:
                raise ValueError(f'Failed checksum: {storage_path}, got {file_checksum} expected {expected_checksum}')
