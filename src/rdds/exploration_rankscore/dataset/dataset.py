from typing import List, Dict, Any, Set, Type

import h5py
from h5py import File, string_dtype
import numpy as np

#from cyvcf2 import VCFReader
from rdds.lib.vcf import VCFReader, Variant
from rdds.lib.vcf import ParsableVariant
from .class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT
from sys import stderr
from pprint import pprint

# Types definitions
info_meta_field = Dict[str, str]
InfoFormatDict: Dict[str, info_meta_field] = {}

class Dataset:

    """
    Compiles a VCF file into a .hdf5 binary dataset file
    """

    def __init__(self,
                 file_path: str):
        """
        Args:
            file_path: Path to dataset
        """
        self.file_path: str = file_path

    @staticmethod
    def get_info_format_fields(vcf_files: List[str]) -> InfoFormatDict:
        """
        Determine the INFO, FORMAT fields available in the VCFs
        :param vcf_files:
        :return:
            Meta data on INFO and FORMAT fields in a InfoFormatDict type
        """

        info_format_fields: InfoFormatDict = {}
        #
        for vcf_file in vcf_files:
            vcf_reader: VCFReader = VCFReader(vcf_file, 'r')

            # Identify and store information on all INFO fields in VCF header
            for entry in vcf_reader.header_iter():
                # Select data only from INFO or FORMAT fields
                if not (entry.type == 'INFO' or entry.type == 'FORMAT'):
                    continue
                # Store metadata
                if entry['ID'] not in info_format_fields.keys():
                    # Add if not present
                    info_format_fields.update({entry['ID']: entry})
                else:
                    # If already present; for every attribute, make sure it's matching with previous observed meta data
                    for key, value in entry:
                        if not info_format_fields[entry['ID']][key] == value:
                            raise TypeError(f'Metadata mismatch; tried to add VCF attribute but the format is conflicting:'
                                            f'{info_format_fields[entry["ID"]]}, {entry}')
            vcf_reader.close()
        return info_format_fields

    @staticmethod
    def resize_dataset(hd5_group: h5py.Group,
                       increment_size: int) -> h5py.Group:
        """
        Resize hd5py dataset outermot dimension
        :param hd5_group: A HD5py Group containing sub datasets
        :param increment_size: Increment by this much
        :return: Bigger dataset
        """
        for dset_name in hd5_group.keys():
            feature_dset = hd5_group[dset_name]
            current_size: List = list(feature_dset.shape)
            current_size[0] += increment_size  # Increase outermost dimension
            feature_dset.resize(tuple(current_size))
        return hd5_group

    @staticmethod
    def add_subset(dataset: h5py.Group,
                   name: str,
                   initial_size: int,
                   dtype: Type,
                   maximum_character_limit: int):
        """
        Add a sub dataset to GROUP.
        :param dataset: Dataset group
        :param name: Name of dataset (i.e. the key)
        :param dtype: Data type
        :return: dataset for chaining
        """
        _dtype = None
        if dtype == str:
            _dtype = (string_dtype('utf-8', maximum_character_limit))
            fillvalue = b'\0'  # String termination
        elif dtype in [float, int, np.float32]:
            _dtype = np.float32
            fillvalue = np.nan  # NaN
        else:
            raise ValueError(f'Unknown dtype Type {dtype}')
        dataset.create_dataset(name=name,
                               shape=(initial_size, 1),
                               maxshape=(None, 1),
                               dtype=_dtype,
                               fillvalue=fillvalue)
        return dataset

    def compile_structured_format(self,
                                  vcf_files: List[str],
                                  features: List[str] = None,
                                  features_ignore: List[str] = None,
                                  maximum_character_limit: int = 128):
        """
        Args:
            vcf_files: List of variant files to be used for dataset creation
            features: The INFO/FORMAT fields to add to dataset (matching VCF field names, e.g. CLINVAR_CLNSIG)
            features_ignore: List of INFO/FORMAT fields to ignore
            maximum_character_limit: Size of str arrays in dataset
        """

        dset_file: File = File(name=self.file_path,
                               mode='w')

        info_format_fields = self.get_info_format_fields(vcf_files)

        # If no explicit features are set, use all available in VCF
        if features is None:
            features = sorted([key for key in info_format_fields.keys()])

        # Default ignored features
        if features_ignore is None:
            features_ignore = []

        # Drop ignore features
        for drop_feature in features_ignore:
            try:
                features.remove(drop_feature)
            except KeyError:
                pass

        # Check that requested INFO, FORMAT fields available in dataset
        for field_name in features:
            if field_name not in info_format_fields.keys():
                raise KeyError(f'VCF data field {field_name} not present in VCF(s)')

        # Setup a dataset with dimensions [n_variants, feature_dimension]
        structured_vcfs: h5py.Group = dset_file.create_group('structured_vcfs')

        # Create VCF index
        structured_vcfs.create_dataset(name='chrom',
                                       shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=(string_dtype('utf-8', maximum_character_limit)),
                                       fillvalue=b'\0')
        structured_vcfs.create_dataset(name='pos',
                                       shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=np.int64,
                                       fillvalue=np.nan)
        structured_vcfs.create_dataset(name='variant_ids',
                                       shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=(string_dtype('utf-8', maximum_character_limit)),
                                       fillvalue=b'\0')
        structured_vcfs.create_dataset(name='ref',
                                       shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=(string_dtype('utf-8', maximum_character_limit)),
                                       fillvalue=b'\0')
        structured_vcfs.create_dataset(name='alt',
                                       shape=(0, 1),
                                       maxshape=(None, 1),
                                       dtype=(string_dtype('utf-8', maximum_character_limit)),
                                       fillvalue=b'\0')

        n_total_variants: int = 0
        row: int = 0
        for vcf_file in vcf_files:
            print(f'Processing VCF: {vcf_file}')
            vcf_reader: VCFReader = VCFReader(vcf_file, 'r')
            n_variants: int = vcf_reader.number_of_variants
            n_total_variants += n_variants

            structured_vcfs = self.resize_dataset(hd5_group=structured_vcfs,
                                                  increment_size=n_variants)
            dset_file.flush()

            try:
                # Parse CSQ field according to Description= .... Format: Allele|consequence|...
                vep_csq_description: str = info_format_fields['CSQ']['Description']
            except AttributeError:
                vep_csq_description = None
            for variant in vcf_reader:
                parsed_variant: ParsableVariant = ParsableVariant(variant,
                                                                  parse_only_fields=features,
                                                                  vep_csq_description=vep_csq_description)

                structured_vcfs['chrom'][row, 0] = parsed_variant.chrom
                structured_vcfs['pos'][row, 0] = parsed_variant.pos
                structured_vcfs['variant_ids'][row, 0] = parsed_variant.id
                structured_vcfs['ref'][row, 0] = parsed_variant.ref
                structured_vcfs['alt'][row, 0] = parsed_variant.alt

                # Add every parsed feature to dataset
                for parsed_field_name in parsed_variant:
                    # If a dataset feature dimension is missing in dataset, add it
                    if parsed_field_name not in structured_vcfs.keys():
                        dtype = type(parsed_variant.__getattribute__(parsed_field_name))
                        structured_vcfs = self.add_subset(dataset=structured_vcfs,
                                                          name=parsed_field_name,
                                                          initial_size=n_total_variants,
                                                          dtype=dtype,
                                                          maximum_character_limit=maximum_character_limit)
                    try:
                        data = parsed_variant.__getattribute__(parsed_field_name)
                        structured_vcfs[parsed_field_name][row, 0] = data
                    except (ValueError) as error:
                        print(f'Failed adding data to dset, {parsed_field_name}: {error}', file=stderr)
                row += 1
                if row % 1000 == 0:
                    print(100 * (float(row) / float(n_total_variants)))

            vcf_reader.close()
        dset_file.flush()
        dset_file.close()

        print(f'Dataset {self.file_path} generation complete')

    def compile_structured_format_mutacc_tp_cases(self, mutacc_vcf_file_path: str):
        """
        Add a truth label in column 'label' as 0: TN/ unconfirmed, 1: TP, confirmed pathogenic variant.

        If there's a matching variant in this (previously compiled) dataset, consider this
        variant also duplicated in MUTACC VCF file as pathogenic by setting label == 1.0.

        Variants in this datafile (self.file_path) that's not duplicated in MUTACC will be considered TN.

        :param mutacc_vcf_file_path: File path to mutacc VCF dump file.
        (All variants in this file is considered TP pathogenic variants).
        """
        with File(self.file_path, 'r+') as dset_file:
            # Since the amount of TP cases are small it's affordable to keep them in RAM for processing speed
            tp_variants: List[ParsableVariant] = []
            vcf_reader: VCFReader = VCFReader(mutacc_vcf_file_path)
            for variant in vcf_reader:
                tp_variants.append(ParsableVariant(variant))
            structured_vcfs = dset_file['structured_vcfs']
            dlen = structured_vcfs['variant_ids'].shape[0]  # outermost dimension is n_variants
            structured_vcfs.create_dataset(name='label',
                                           shape=(dlen, 1),
                                           maxshape=(dlen, 1),
                                           dtype=np.float32,
                                           fillvalue=LABEL_BENIGN_VARIANT)
            for idx in range(0, dlen):
                # To be able to compare variant attributes, datatypes must be identical (python standard dtypes)
                chrom = (structured_vcfs['chrom'][idx, 0]).decode('utf-8')
                pos = int(structured_vcfs['pos'][idx, 0])
                ref = (structured_vcfs['ref'][idx, 0]).decode('utf-8')
                alt = (structured_vcfs['alt'][idx, 0]).decode('utf-8')
                # For every variant in MUTACC set, mark variant as causative if found in existing dataset
                for disease_causing_variant in tp_variants:
                    if disease_causing_variant.chrom == chrom and \
                       disease_causing_variant.pos == pos and \
                       disease_causing_variant.ref == ref and \
                       disease_causing_variant.alt == alt:
                        structured_vcfs['label'][idx, 0] = LABEL_PATHOGENIC_VARIANT  # Set variant label as disease causing
            n_tps: float = np.sum(structured_vcfs["label"][:, 0])
            tp_ratio: float = n_tps / float(dlen)
            if tp_ratio == 0:
                raise ValueError(f'Expected some ratio of TPs in dataset, found none')
            print(f'Ratio of TP variants in dataset: {tp_ratio}, dropped \
{(len(tp_variants) - n_tps) / len(tp_variants)} of MUTACC true positive causative variants')
            dset_file.flush()
            dset_file.close()
            print('Addition of MUTACC variants complete.')

    def compile(self,
                vcf_files: List[str],
                features: List[str] = None,
                vcf_file_mutacc_confirmed_tp_cases: str = None):
        self.compile_structured_format(vcf_files=vcf_files,
                                       features=features)
        if vcf_file_mutacc_confirmed_tp_cases is not None:
            self.compile_structured_format_mutacc_tp_cases(mutacc_vcf_file_path=vcf_file_mutacc_confirmed_tp_cases)

    def view(self):
        """
        Print dataset meta info to stdout such as groups, shapes and dtypes.
        :return:
        """

        def print_dataset(name: str, group: h5py.Group):
            s = f'{self.file_path}::{name}: '
            def add_attrib(attrib_name):
                nonlocal s
                try:
                    s += f'{attrib_name}={group.__getattribute__(attrib_name)} '
                except AttributeError:
                    pass
            add_attrib('shape')
            add_attrib('dtype')
            print(s)

        with File(self.file_path, 'r') as h5py_file:
            h5py_file.visititems(print_dataset)
