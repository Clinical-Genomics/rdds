from cyvcf2 import Variant
import numpy as np
from typing import *
from collections import Iterator

_NUMERICAL_DATA_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

_VCF_BASE_HEADER: List[str] = ['CHROM', 'POS', 'ID', 'REF', 'ALT']

_NESTED_NAME_TO_POSITION: Dict[str, int] = {  # Dictionary mapping field name to position in VCF INFO sub field
    'RankScore_family_id': 0,
    'RankScore_value': 1,
    'RankScoreNormalized_family_id': 0,
    'RankScoreNormalized_value': 1,
    'RankScoreMinMax_family_id': 0,
    'RankScoreMinMax_min': 1,
    'RankScoreMinMax_max': 2,
    'Compounds_family_id': 0,
    'Compounds_value': 1,
    'GeneticModels_family_id': 0,
    'GeneticModels_model': 1,
    'ModelScore_family_id': 0,
    'ModelScore_value': 1,
}


class ParsingError(Exception):
    """
    Base class for parsing related errors.
    """


class DtypeNotFoundError(ParsingError):
    """
    Data type not found for field name error
    """
    pass


class NotExpectedDtypeError(ParsingError):
    """
    Data type does not conform to expected dtype
    """
    pass


class ParsableVariant:

    """
    This class implements wrappers on top of cyvcf2.Variant to allow
    parsing of nested INFO, FORMAT fields in a VCF variant entry.

    This class contains field specific datatype declarations since nested INFO/FORMAT fields
    does not allow for specifying innermost data types but just the overall data format
    (if multiple sub entries).
    """

    def __init__(self,
                 variant: Variant,
                 parse_only_fields: List[str] = [],
                 vep_csq_description: str = None):
        """
        :param variant: A cyvcf2 Variant instance
        :param parse_only_fields: List of INFO fields to parse
        :param vep_csq_description: CSQ.DESCRIPTION field containing 'Format: 'Allele|Consequence|...' string

        TODO: Parse FORMAT fields
        """
        if not isinstance(variant, Variant):
            raise TypeError(f'Expected cyvcf2.Variant got {type(self)}')

        self._vep_csq_description: str = vep_csq_description

        # Store all meta data as private instance attributes
        self.CHROM = str(variant.CHROM)  # chrom idx + x, y, mit as char
        self.POS = int(variant.POS)  # integer
        self.ID = str(variant.ID)  # text field
        # TODO: Parsing of REF/ALT fields with <ID> subtype, see https://samtools.github.io/hts-specs/VCFv4.1.pdf, p4
        self.REF: str = str(variant.REF).replace('[', '').replace(']', '')
        self.ALT: str = str(variant.ALT).replace('[', '').replace(']', '')
        self._parsed_fields: List[str] = _VCF_BASE_HEADER.copy()  # Placeholder of actually parsed values
        for field_name, field_value in list(variant.INFO):
            if len(parse_only_fields) > 0 and field_name not in parse_only_fields:
                continue
            parsing_fn: callable = self._parse_store_data  # fallback parsing function
            try:  # custom parser
                parsing_fn = vars(self)['_parse_fn_%s' % field_name]
            except KeyError:
                pass
            parsing_fn(key=field_name, data=field_value)

    def __iter__(self) -> Iterator:
        """
        Return parsed fields as Iterator
        :return:
        """
        return iter(self._parsed_fields)

    """
    Below follows custom parsing functions for INFO, FORMAT fields.
    
    The functions should be providing the following:
    * Name of function should map to the VCF.INFO.[FIELD_NAME], case sensitive, prefixed with _parse_fn
    * Parsing function should set class attributes with the following spec:
        * self.[FIELD_NAME]_[SUFFIX] -> the_data: Any
    * Should add parsed fields into self._parsed_fields

    Note: All text strings are treated case-sensitive!
    
    TODO: Check the Type INFO field for dtype sanity check, info_format_fields[field_name]['Type']
    """

    def get_dtype(self, field_name: str) -> Type:
        """
        Return parsed data type for a VCF field.
        :param field_name: VCF field name (parsed)
        :return: Type
        :raises DtypeNotFoundError: If field name was not found
        """
        try:
            return type(vars(self)[field_name])
        except KeyError:
            raise DtypeNotFoundError(f'Field name \'{field_name}\' not found')

    def _parse_store_data(self,
                          key: str,
                          data: Any,
                          expected_dtype: Type = None):
        """
        Parse and store data in instance.

        Always try to parse the data into float if possible.
        If the data contains alphanumerics, treat it as strings.
        Throws ParsingError on special characters in data.

        :param key: Instance attribute key where data is to be stored
        :param data: Data object
        :param expected_dtype: Expected type, throws NotExpectedDtypeError if
          the data is not sucesssfully parsed into this dtype.
        :return:
        """

        if not isinstance(data, (str, int, float)):
            raise ValueError(f'Unknown data type: {type(data)}:{data}')

        def parse_numerical_data(data: Union[float, int]) -> float:
            return float(data)

        def parse_text_data(data: str) -> Union[float, str]:
            # Cases: empty string, numeric or alpha string content
            # Using str.isnumerical() does not work for negative numbers. Use try-catch instead.
            try:
                data_parsed: float = float(data)
            except ValueError:
                if data.isalpha():
                    data_parsed: str = data
                elif data.isprintable():  # all printable alphanumerical characters
                    data_parsed: str = data
                else:
                    raise ParsingError(f'Data contains invalid characters {key}={data}')
            return data_parsed

        if isinstance(data, (float, int)):
            data_parsed = parse_numerical_data(data)
        elif isinstance(data, str):
            data_parsed = parse_text_data(data)
        else:
            raise ParsingError(f'Failed to parse data {data}')

        if expected_dtype is not None and not isinstance(data_parsed, expected_dtype):
            raise NotExpectedDtypeError(f'{type(data_parsed)}:{data_parsed} != {expected_dtype}')

        if key in self._parsed_fields:  # Sanity check
            raise ValueError(f'Key \'{key}\' is already parsed in variant {self._parsed_fields}')

        self.__setattr__(key, data_parsed)
        self._parsed_fields.extend([key])

    @property
    def parsed_fields(self) -> List[str]:
        """
        Return the names of all parsed fields.
        :return:
        """
        return self._parsed_fields

    def _parse_fn_RankScore(self,
                            text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='RankScore_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScore_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='RankScore_value',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScore_value']],
                               expected_dtype=float)

    def _parse_fn_RankScoreNormalized(self,
                                      text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='RankScoreNormalized_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScoreNormalized_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='RankScoreNormalized_value',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScoreNormalized_value']],
                               expected_dtype=float)

    def _parse_fn_RankScoreMinMax(self,
                                  text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='RankScoreMinMax_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScoreMinMax_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='RankScoreMinMax_min',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScoreMinMax_min']],
                               expected_dtype=float)
        self._parse_store_data(key='RankScoreMinMax_max',
                               data=parts[_NESTED_NAME_TO_POSITION['RankScoreMinMax_max']],
                               expected_dtype=float)

    def _parse_fn_RankResult(self,
                             text: str):
        self._parse_store_data(key='RankResult', data=text)

    def _parse_fn_Compounds(self,
                            text: str):
        parts: List[str] = text.split(':')
        if len(parts) != 2:
            raise ValueError('Expected len of two')
        self._parse_store_data(key='Compounds_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['Compounds_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='Compounds_value',
                                   data=parts[_NESTED_NAME_TO_POSITION['Compounds_value']],
                                   expected_dtype=str)

    def _parse_fn_GeneticModels(self,
                             text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='GeneticModels_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['GeneticModels_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='GeneticModels_model',
                               data=parts[_NESTED_NAME_TO_POSITION['GeneticModels_model']],
                               expected_dtype=str)

    def _parse_fn_ModelScore(self,
                             text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='ModelScore_family_id',
                               data=parts[_NESTED_NAME_TO_POSITION['ModelScore_family_id']],
                               expected_dtype=str)
        self._parse_store_data(key='ModelScore_value',
                               data=parts[_NESTED_NAME_TO_POSITION['ModelScore_value']],
                               expected_dtype=float)

    def _parse_fn_MitomapSomaticMutations(self,
                                          text: str):
        # TODO: Split on & char
        self._parse_store_data(key='MitomapSomaticMutations',
                               data=self._hack_bugfix_ensemble_vep_430(text))

    def _parse_fn_CSQ(self,
                      text: str):
        """
        # TODO: Missing, NAN values represented as '-', don't store it as string if it's observed
        Parse CSQ INFO field according to vep_csq_format, e.g. 'Format: Allele|Consequence|IMPACT|SYMBOL|Gene|...'
        :param text: String containing format spec
        :return: parsed CSQ fields as dict
        """
        if self._vep_csq_description is None:
            return {}
        keys = self._vep_csq_description.split('Format: ')[1].replace('"', '').split('|')
        keys = ['CSQ_%s' % key for key in keys]
        data = text.split('|')
        for key, value in zip(keys, data):
            if key == 'CSQ_HGVSp':
                self._parse_store_data(key=key,
                                       data=self._hack_bugfix_ensemble_vep_430(value))
            else:
                self._parse_store_data(key=key, data=value)

    def get_attribute(self,
                      key: str,
                      preferred_dtype: Type = None) -> Any:
        """
        Wraps __getattr__, vars() to provide typesafe API for float and str data
        :param key:
        :return:
        """
        try:
            data: Any = vars(self)[key]
        except KeyError:
            # Attribute was not parsed, return NaN
            return np.NaN
        if isinstance(data, str) and data == '' and preferred_dtype in _NUMERICAL_DATA_TYPES:
            return np.NaN
        return data

    @staticmethod
    def _hack_bugfix_ensemble_vep_430(text: str) -> str:
        """
        Bugfix: Data from VEP HGSVp contains '%3D', '=' encoded
        https://github.com/Ensembl/ensembl-vep/issues/430

        Furthermore, in MitomapSomaticMutations %2C encoded comma has been observed.

        :text: Input string that contains bad data
        :return: Text with bad data removed
        """
        return text.replace('%3D', '') \
                   .replace('%2C', '')
