from urllib.request import urlretrieve
from os.path import join, basename
from os import listdir, remove

from . import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.checksum import checksum

# Mouse Genome Informatics https://www.informatics.jax.org/mgihome/projects/aboutmgi.shtml
_JAX_SRC = 'http://www.informatics.jax.org/downloads/reports/'

_LOGGER = get_logger('dataset-mouse-human', 'info')
class MouseHuman:

    def __init__(self):
        # Genotypes and Mammalian Phenotype Annotations for Marker Type Genes excluding conditional mutations
        # https://www.informatics.jax.org/downloads/reports/index.html#pheno
        self._mouse_phenotypes_to_genes = _JAX_SRC + '/MGI_GenePheno.rpt'
        self._mouse_phenotypes_to_genes_sha256 = None

        self._files_checksums = [
            (self._mouse_phenotypes_to_genes, self._mouse_phenotypes_to_genes_sha256)
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

