from urllib.request import urlretrieve
from os.path import join, basename
from os import listdir, remove

from . import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.checksum import checksum

_HPO_SRC='https://github.com/obophenotype/human-phenotype-ontology/releases/download/'
_HPO_VERSION = 'v2026-01-08'

_LOGGER = get_logger('dataset-hpo', 'info')
class HPO:

    def __init__(self):
        # https://github.com/obophenotype/human-phenotype-ontology/blob/v2026-01-08/docs/annotations/phenotype_to_genes.md
        self._phenotype_to_genes = _HPO_SRC + _HPO_VERSION + '/phenotype_to_genes.txt'
        self._phenotype_to_genes_sha256 = '3ce4be69b16e2257b52e377e6248ced09b876d458adfe60b162aeaad99522711'

        self._files_checksums = [
            (self._phenotype_to_genes, self._phenotype_to_genes_sha256)
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

