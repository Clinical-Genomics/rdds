import cyvcf2
from cyvcf2 import Variant
from typing import *
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

    def _get_type_fields(self, type: str) -> List[str]:
        """
        Return VCF field ID (names) of fields in category 'type'
        :param type:
        :return:
        """
        return [entry['ID'] for entry in list(self.header_iter()) if entry['HeaderType'] == type]

    @property
    def static_data_fields(self) -> List[str]:
        """
        Static fields that's always expected to be present in VCF
        :return:
        """
        return ['CHROM', 'POS', 'ID', 'REF', 'ALT']

    @property
    def format_fields(self) -> List[str]:
        """
        Return all FORMAT fields
        :return:
        """
        return self._get_type_fields(type='FORMAT')

    @property
    def info_fields(self) -> List[str]:
        """
        Return all INFO fields
        :return:
        """
        return self._get_type_fields(type='INFO')

    @property
    def data_fields(self) -> List[str]:
        """
        Return all fields containing variant data
        :return:
        """
        data_fields: List[str] = self.static_data_fields
        data_fields.extend(self.info_fields)
        data_fields.extend(self.format_fields)
        return data_fields

    @property
    def csq_description(self) -> Union[str, None]:
        """
        Return CSQ Description field content
        :return:
        """
        for header in self.header_iter():
            try:
                if header['ID'] == 'CSQ':
                    vep_csq_description: str = header['Description']
                    return vep_csq_description
            except (AttributeError, KeyError):
                pass

    @property
    def csq_sub_fields(self) -> Union[List[str], None]:
        """
        Return CSQ Description sub field names as list
        :return:
        """
        vep_csq_description = self.csq_description
        if vep_csq_description is None:
            return
        keys = vep_csq_description.split('Format: ')[1].replace('"', '').split('|')
        keys = ['CSQ_%s' % key for key in keys]
        return keys