from cyvcf2 import Variant
import numpy as np
from typing import *
from collections import Iterator

_NUMERICAL_DTYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

class ParsingError(Exception):
    """
    Parsing of data failed error
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


class NoParserFoundError(ParsingError):
    """
    Raised when no custom parser was found for data
    """

class VcfFieldNotFoundError(ParsingError):
    """
    Raised when the VCF field was not found (in this variant)
    """


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
                 parse_only_fields: List[str] = None,
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
        self.CHROM: str = str(variant.CHROM)  # chrom idx + x, y, mit as char
        self.POS: int = int(variant.POS)  # integer
        self.ID: str = str(variant.ID)  # text field
        # TODO: Parsing of REF/ALT fields with <ID> subtype, see https://samtools.github.io/hts-specs/VCFv4.1.pdf, p4
        self.REF: str = str(variant.REF).replace('[', '').replace(']', '')
        self.ALT: str = str(variant.ALT).replace('[', '').replace(']', '')
        self._parsed_fields: List[str] = ['CHROM', 'POS', 'ID', 'REF', 'ALT']  # Placeholder of actually parsed values
        for key, value in list(variant.INFO):

            if key not in parse_only_fields:
                continue

            try:
                self._parse_store_data(key, value)
            except ParsingError:
                try:  # custom parser
                    parsing_fn: callable = self.__getattribute__('_parse_fn_%s' % key)
                except AttributeError:
                    raise NoParserFoundError(f'Found no parser for field \'{key}\', contains \'{value}\'')
                if parsing_fn is not None:
                    parsing_fn(value)

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
        """
        try:
            return type(self.__getattribute__(field_name))
        except AttributeError:
            raise DtypeNotFoundError(f'Field name \'{field_name}\' not found')

    def _parse_store_data(self,
                          key: str,
                          data: Any,
                          expected_dtype: Type = None,
                          allowed_special_characters: Set[str] = None):
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

        if allowed_special_characters is None:
            allowed_special_characters: Set[str] = set()

        if isinstance(data, (float, int)):
            data_parsed: float = float(data)
        elif isinstance(data, str):
            # Cases: empty string, numeric or alpha string content
            # Using str.isnumerical() does not work for negative numbers. Use try-catch instead.
            try:
                data_parsed: float = float(data)
            except ValueError:
                if data.isalpha():
                    data_parsed: str = data
                elif data.isprintable():  # all printable alphanumerical characters
                    # Check if additional special characters allowed
                    nonalphanumerics: Set[str] = set([c for c in data if not c.isalnum()])
                    allowed_diff = nonalphanumerics.difference(allowed_special_characters)
                    if len(allowed_diff) > 0:
                        raise ParsingError(f'Data field \'{key}\' contains non-allowed data: \'{data}\', {allowed_diff}')
                    data_parsed: str = data
                else:
                    raise ParsingError(f'Data contains invalid characters {key}={data}')
        else:
            raise ParsingError(f'Failed to parse data {data}')

        if expected_dtype is not None and not isinstance(data_parsed, expected_dtype):
            raise NotExpectedDtypeError(f'{type(data_parsed)}:{data_parsed} != {expected_dtype}')

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
        self._parse_store_data(key='RankScore_family_id', data=parts[0], expected_dtype=str)
        self._parse_store_data(key='RankScore_value', data=parts[1], expected_dtype=float)

    def _parse_fn_RankScoreNormalized(self,
                                      text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='RankScoreNormalized_family_id', data=parts[0], expected_dtype=str)
        self._parse_store_data(key='RankScoreNormalized_value', data=parts[1], expected_dtype=float)

    def _parse_fn_RankScoreMinMax(self,
                                  text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='RankScoreMinMax_family_id', data=parts[0], expected_dtype=str)
        self._parse_store_data(key='RankScoreMinMax_min', data=parts[1], expected_dtype=float)
        self._parse_store_data(key='RankScoreMinMax_max', data=parts[2], expected_dtype=float)

    def _parse_fn_RankResult(self,
                             text: str):
        parts: List[str] = text.split('|')
        for i, sub_text in enumerate(parts):
            self._parse_store_data(key='RankResult-%d' % i, data=sub_text)

    def _parse_fn_Compounds(self,
                            text: str):
        parts: List[str] = text.split(':')
        self._parse_store_data(key='Compounds_family_id', data=parts[0], expected_dtype=str)
        for i, compound in enumerate(parts[1].split('|')):
            self._parse_store_data(key='Compounds_value-%d' % i,
                                   data=compound,
                                   expected_dtype=str,
                                   allowed_special_characters={'_', '-', '>'})

    def _parse_fn_GeneticModels(self,
                             text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='GeneticModels_family_id',
                               data=parts[0],
                               expected_dtype=str)
        self._parse_store_data(key='GeneticModels_model',
                               data=parts[1],
                               allowed_special_characters={'_', '|'},
                               expected_dtype=str)

    def _parse_fn_ModelScore(self,
                             text: str):
        parts: list = text.split(':')
        self._parse_store_data(key='ModelScore_family_id', data=parts[0], expected_dtype=str)
        self._parse_store_data(key='ModelScore_value', data=parts[1], expected_dtype=float)

    def _parse_fn_MitomapAssociatedDiseases(self,
                                            text: str):
        # TODO: Split on disease names, & char
        self._parse_store_data(key='MitomapAssociatedDiseases', data=text, allowed_special_characters={'-', '.', '_', '/', '&'})

    def _parse_fn_OMIM(self,
                       text: str):
        self._parse_store_data(key='OMIM', data=text, allowed_special_characters={'.', '#'})

    def _parse_fn_AaVarH(self,
                         text: str):
        self._parse_store_data(key='AaVarH', data=text, allowed_special_characters={'.'})

    def _parse_fn_AaVarP(self,
                         text: str):
        self._parse_store_data(key='AaVarP', data=text, allowed_special_characters={'.'})

    def _parse_fn_MitomapSomaticMutations(self,
                                          text: str):
        # TODO: Split on & char
        self._parse_store_data(key='MitomapSomaticMutations',
                               data=self._hack_bugfix_ensemble_vep_430(text),
                               allowed_special_characters={'/', '_', '-', '.', '&'})

    def _parse_fn_MutPred_Probability(self,
                                      text: str):
        self._parse_store_data(key='MutPred_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_MutPred_Prediction(self,
                                     text: str):
        self._parse_store_data(key='MutPred_Prediction', data=text, allowed_special_characters={'.', '_'})

    def _parse_fn_Polyphen2HumDiv_Prediction(self,
                                             text: str):
        self._parse_store_data(key='Polyphen2HumDiv_Prediction', data=text, allowed_special_characters={'.', '_'})

    def _parse_fn_Polyphen2HumVar_Prediction(self,
                                             text: str):
        self._parse_store_data(key='Polyphen2HumDiv_Prediction', data=text, allowed_special_characters={'.'})

    def _parse_fn_Polyphen2HumDiv_Probability(self,
                                              text: str):
        self._parse_store_data(key='Polyphen2HumDiv_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_Polyphen2HumVar_Probability(self,
                                              text: str):
        self._parse_store_data(key='Polyphen2HumVar_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_PhDSNP_Prediction(self,
                                    text: str):
        self._parse_store_data(key='PhDSNP_Prediction', data=text, allowed_special_characters={'.'})

    def _parse_fn_PhDSNP_Probability(self,
                                    text: str):
        self._parse_store_data(key='PhDSNP_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_SNPsGO_Probability(self,
                                     text: str):
        self._parse_store_data(key='SNPsGO_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_SNPsGO_Prediction(self,
                                     text: str):
        self._parse_store_data(key='SNPsGO_Prediction', data=text, allowed_special_characters={'.'})

    def _parse_fn_dbSNP(self,
                        text: str):
        self._parse_store_data(key='dbSNP', data=text, allowed_special_characters={'.'})

    def _parse_fn_dbSNP_Probabiltity(self,
                        text: str):
        self._parse_store_data(key='dbSNP_Probabiltity', data=text, allowed_special_characters={'.'})

    def _parse_fn_AaChange(self,
                        text: str):
        self._parse_store_data(key='AaChange', data=text, allowed_special_characters={'.'})

    def _parse_fn_DiseaseScore(self,
                               text: str):
        self._parse_store_data(key='DiseaseScore', data=text, allowed_special_characters={'.'})

    def _parse_fn_Panther_Prediction(self,
                                      text: str):
        self._parse_store_data(key='Panther_Prediction', data=text, allowed_special_characters={'.'})

    def _parse_fn_Panther_Probability(self,
                                      text: str):
        self._parse_store_data(key='Panther_Probability', data=text, allowed_special_characters={'.'})

    def _parse_fn_Clinvar(self,
                          text: str):
        self._parse_store_data(key='Clinvar', data=text, allowed_special_characters={'.'})

    def _parse_fn_Pathogenicity(self,
                                text: str):
        self._parse_store_data(key='Pathogenicity', data=text, allowed_special_characters={'.', '_'})

    def _parse_fn_Locus(self,
                        text: str):
        self._parse_store_data(key='Locus', data=text, allowed_special_characters={'-', '.'})

    def _parse_fn_1KGenomesHeteroplasmy(self,
                                        text: str):
        self._parse_store_data(key='1KGenomesHeteroplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_1KGenomesHomoplasmy(self,
                                      text: str):
        self._parse_store_data(key='1KGenomesHomoplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_MitomapHomoplasmy(self,
                                    text: str):
        self._parse_store_data(key='MitomapHomoplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_MitomapHeteroplasmy(self,
                                      text: str):
        self._parse_store_data(key='MitomapHeteroplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP_AM(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqP_AM', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP_EU(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqP_EU', data=text, allowed_special_characters={'.'})

    def _parse_fn_SomaticMutationsHomoplasmy(self,
                                            text: str):
        self._parse_store_data(key='SomaticMutationsHomoplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_SomaticMutationsHeteroplasmy(self,
                                               text: str):
        self._parse_store_data(key='SomaticMutationsHeteroplasmy', data=text, allowed_special_characters={'.'})

    def _parse_fn_NtVarH(self,
                         text: str):
        self._parse_store_data(key='NtVarH', data=text, allowed_special_characters={'.'})

    def _parse_fn_NtVarP(self,
                         text: str):
        self._parse_store_data(key='NtVarP', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP(self,
                              text: str):
        self._parse_store_data(key='AlleleFreqP', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH(self,
                              text: str):
        self._parse_store_data(key='AlleleFreqH', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH_AM(self,
                               text: str):
        self._parse_store_data(key='AlleleFreqH_AM', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH_OC(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqH_OC', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH_EU(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqH_EU', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH_AF(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqH_AF', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqH_AS(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqH_AS', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP_AF(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqP_AF', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP_OC(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqP_OC', data=text, allowed_special_characters={'.'})

    def _parse_fn_AlleleFreqP_AS(self,
                                 text: str):
        self._parse_store_data(key='AlleleFreqP_AS', data=text, allowed_special_characters={'.'})

    def _parse_fn_HmtVar(self,
                         text: str):
        self._parse_store_data(key='HmtVar', data=text, allowed_special_characters={'.'})

    def _parse_fn_mitotip_trna_prediction(self,
                         text: str):
        self._parse_store_data(key='mitotip_trna_prediction', data=text, allowed_special_characters={'_'})

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
            if key == 'CSQ_genomic_superdups_frac_match':
                # Nested sub field
                for i, sub_value in enumerate(value.split('&')):
                    self._parse_store_data(key=key+'-%d' % i, data=sub_value)
            elif key == 'CSQ_CLINVAR_CLNREVSTAT':
                # TODO: Move this to bugfix method
                value = value.replace(',', '')  # ',-' data observed in this field
                for i, revision_status in enumerate(value.split('&')):
                    self._parse_store_data(key=key+'-%d' % i,
                                           data=revision_status,
                                           allowed_special_characters={'-', '_'})
            elif key == 'CSQ_CLINVAR_CLNSIG':
                for i, sub_value in enumerate(value.split('&')):
                    self._parse_store_data(key=key + '-%d' % i, data=sub_value, allowed_special_characters={'/', '_'})
            elif key in ['CSQ_EXON', 'CSQ_INTRON']:
                for i, sub_value in enumerate(value.split('/')):
                    self._parse_store_data(key=key + '-%d' % i, data=sub_value)
            elif key == 'CSQ_Codons':
                for i, sub_value in enumerate(value.split('/')):
                    self._parse_store_data(key=key + '-%d' % i, data=sub_value, allowed_special_characters={'-'})
            elif key == 'CSQ_HGVSc':
                if value != '':
                    hgvsc_reference, hgvsc_description = value.split(':')
                    self._parse_store_data(key=key+'-reference',
                                           data=hgvsc_reference,
                                           allowed_special_characters={'.', '_'})
                    self._parse_store_data(key=key + '-description',
                                           data=hgvsc_description,
                                           allowed_special_characters={'.', '>', '-', '+', '_', '*'})
            elif key == 'CSQ_HGVSp':
                self._parse_store_data(key=key,
                                       data=self._hack_bugfix_ensemble_vep_430(value),
                                       allowed_special_characters={'.', ':', '_', '?'})
            elif key == 'CSQ_DOMAINS':
                # TODO: Split on database names such as PANTHER, SUPERFAMILY rather than on position idx
                # Syntax: DBNAME:INFO&DNMANE:INFO& ...
                for i, sub_value in enumerate(value.split('&')):
                    self._parse_store_data(key=key + '-%d' % i,
                                           data=sub_value,
                                           allowed_special_characters={':', '.', '_', '(', ')', '-'})
            elif key == 'CSQ_Amino_acids':
                self._parse_store_data(key=key, data=value, allowed_special_characters={'/', '-', '*'})
            elif key == 'CSQ_TRANSCRIPTION_FACTORS':
                for i, sub_value in enumerate(value.split('&')):
                    self._parse_store_data(key=key + '-%d' % i,
                                           data=sub_value,
                                           allowed_special_characters={':'})
            elif key in ['CSQ_cDNA_position', 'CSQ_CDS_position', 'CSQ_Protein_position']:
                self._parse_store_data(key=key, data=value, allowed_special_characters={'?', '-'})
            else:
                self._parse_store_data(key=key, data=value, allowed_special_characters={'-', '_', '&', '.'})

    def _parse_fn_most_severe_consequence(self,
                                          text: str):
        """
        Parse INFO most_severe_consequence field.
        Example: '37102:-|upstream_gene_variant,38034:-|downstream_gene_variant'
        :param text:
        :return:
        """
        sub_texts: List[str] = text.split(',')
        key = 'most_severe_consequence'
        for i, sub_text in enumerate(sub_texts):
            numerical_value, rest = sub_text.split(':')
            alt_variant, vep_so_term = rest.split('|')
            self._parse_store_data(key=key + '-num-%d' % i,
                                   data=numerical_value)
            self._parse_store_data(key=key + '-alt-%d' % i,
                                   data=alt_variant,
                                   allowed_special_characters={'-'})
            self._parse_store_data(key=key + '-vep-%d' % i,
                                   data=vep_so_term,
                                   allowed_special_characters={'_'})

    def _parse_fn_Annotation(self,
                             text: str):
        parts: List[str] = text.split(',')
        for i, sub_text in enumerate(parts):
            self._parse_store_data(key='Annotation-%d' % i, data=sub_text, allowed_special_characters={'-'})

    def get_attribute(self,
                      key: str,
                      preferred_dtype: Type = None) -> Any:
        """
        Wraps __getattr__ to provide typesafe API for float and str data
        :param key:
        :return:
        """
        try:
            data: Any = self.__getattribute__(key)
        except AttributeError:
            # Attribute was not parsed, return NaN
            return np.NaN
        if isinstance(data, str) and data == '' and preferred_dtype in _NUMERICAL_DTYPES:
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
