from cyvcf2 import Variant
import numpy as np
from typing import Any, Dict, List
from collections import Iterator


class ParsableVariant:

    """
    This class implements wrappers on top of cyvcf2.Variant to allow
    parsing of nested INFO, FORMAT fields in a VCF variant entry.
    """

    def __init__(self,
                 variant: Variant,
                 parse_only_fields: List[str] = None,
                 vep_csq_description: str = None):
        """
        :param variant: A cyvcf2 Variant instance
        :param parse_only_fields: List of INFO fields to parse
        :param vep_csq_description: CSQ.DESCRIPTION field containing 'Format: 'Allele|Consequence|...' string

        TODO: Support for FORMAT fields
        """
        if not isinstance(variant, Variant):
            raise TypeError(f'Expected cyvcf2.Variant got {type(self)}')

        self.chrom: str = str(variant.CHROM)  # chrom idx + x, y, mit as char
        self.pos: int = int(variant.POS)  # integer
        self.id: str = str(variant.ID)  # text field
        # TODO: Parsing of REF/ALT fields with <ID> subtype, see https://samtools.github.io/hts-specs/VCFv4.1.pdf, p4
        self.ref: str = str(variant.REF).replace('[', '').replace(']', '')
        self.alt: str = str(variant.ALT).replace('[', '').replace(']', '')

        self._vep_csq_description: str = vep_csq_description

        self._parsed_fields: List[str] = []

        if parse_only_fields is None:
            parse_only_fields: List[str] = [key for key, value in list(variant.INFO)]

        # Store all meta data as private instance attributes
        for key, value in variant.INFO:

            if key not in parse_only_fields:
                continue

            # Try parse field as np.float32 by default
            try:
                self.__setattr__(key, np.float32(value))
            except ValueError as error:
                try:  # custom parser, if available
                    parsing_fn: callable = self.__getattribute__('_parse_fn_%s' % key)
                    if parsing_fn is not None:
                        parsing_fn(value)
                except AttributeError as error:
                    # Set private attribute as fallback, to allow runtime inspection
                    self.__setattr__('_unparsed_'+key, value)

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
    * Parsing function should set class attributes with the following spec: self.[FIELD_NAME]_[SUFFIX] -> the_data: Any
    * Should add parsed fields into self._parsed_fields

    Note: All text strings are treated case-sensitive!
    
    TODO: Check the Type INFO field for dtype sanity check, info_format_fields[field_name]['Type']
    """

    def _parse_fn_RankScore(self,
                            text: str):
        parts: list = text.split(':')
        family_id = parts[0]
        rank_score_value = np.float32(parts[1])
        self.__setattr__('RankScore_family_id', family_id)
        self.__setattr__('RankScore_value', rank_score_value)
        self._parsed_fields.extend(['RankScore_family_id', 'RankScore_value'])

    def _parse_fn_RankScoreNormalized(self,
                                      text: str):
        parts: list = text.split(':')
        family_id = parts[0]
        rank_score_value = np.float32(parts[1])
        self.__setattr__('RankScoreNormalized_family_id', family_id)
        self.__setattr__('RankScoreNormalized_value', rank_score_value)
        self._parsed_fields.extend(['RankScoreNormalized_family_id', 'RankScoreNormalized_value'])

    def _parse_fn_RankScoreMinMax(self,
                                  text: str):
        parts: list = text.split(':')
        family_id = parts[0]
        rank_score_min = np.float32(parts[1])
        rank_score_max = np.float32(parts[2])
        self.__setattr__('RankScoreMinMax_family_id', family_id)
        self.__setattr__('RankScoreMinMax_min', rank_score_min)
        self.__setattr__('RankScoreMinMax_max', rank_score_max)
        self._parsed_fields.extend(['RankScoreNormalized_family_id', 'RankScoreMinMax_min', 'RankScoreMinMax_max'])

    def _parse_fn_CSQ(self,
                      text: str):
        """
        Parse CSQ INFO field according to vep_csq_format, e.g. 'Format: Allele|Consequence|IMPACT|SYMBOL|Gene|...'
        :param text: String containing format spec
        :return: parsed CSQ fields as dict

        # FIXME: CSQ_genomic_superdups_frac_match contains multiple float values in array
        """
        if self._vep_csq_description is None:
            return {}
        keys = self._vep_csq_description.split('Format: ')[1].split('|')
        keys = ['CSQ_%s' % key for key in keys]
        data = text.split('|')
        for i, value in enumerate(data):
            try:
                data[i] = np.float32(value)
            except ValueError as error:
                # TODO: Check if numerical value in string, if so then fail
                pass
        parsed: Dict[str, str] = dict(zip(keys, data))
        for key, value in parsed.items():
            self.__setattr__(key, value)
        self._parsed_fields.extend(list(parsed.keys()))
