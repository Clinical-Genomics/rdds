import cyvcf2
from cyvcf2 import Variant

import subprocess as sp
from tempfile import NamedTemporaryFile


class VCFReader(cyvcf2.VCFReader):

    def __init__(self,
                 fname: str,
                 *args,
                 **kwargs):

        self.fname: str = fname

        # Unpack the gzipped file to improve read performance
        if '.gz' in fname:
            self.tmp_file: NamedTemporaryFile = NamedTemporaryFile('r')
            process: sp.Popen = sp.Popen(f'gunzip --keep -c {fname} > {self.tmp_file.name}', shell=True)
            process.wait()
            if not process.returncode == 0:
                raise ValueError(f'Failed to unpack file, got {process.returncode}: {process.stderr}')
            self.fname = self.tmp_file.name

        super().__init__(fname, *args, **kwargs)

        self._number_of_variants: int = None

    @property
    def number_of_variants(self) -> int:
        """
        Compute the number of variants in the VCF file (i.e. rows not prepended with '#')
        :return: number of rows in file
        """
        if self._number_of_variants is None:
            # Count rows containing # as header rows
            header_rows: int = 0
            with open(self.fname, "rbU") as f:
                for line in f:
                    if line[0] == 35:  # binary value for ascii '#' character
                        header_rows += 1
                    else:
                        break
            # Count total amount of rows in file in a fast way, subtract header rows
            with open(self.fname, "rbU") as f:
                self._number_of_variants = sum(1 for _ in f) - header_rows

        return self._number_of_variants
