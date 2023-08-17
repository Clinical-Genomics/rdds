import re
from typing import *
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta
import os
import gc
import h5py
from h5py import File as Hd5File, string_dtype, Dataset as Hd5DataSet, Group as Hd5Group
import numpy as np
from multiprocessing import Queue
from rdds.lib.processpool import ProcessPool, DummyPool, Task, CompletedTaskQueue, MULTIPROCESSING_LOGGER
from dataclasses import dataclass
from re import match, Match
from os import remove

from rdds.lib.resource_usage import ProcessResourceUsage
from rdds.lib.vcf import VCFReader, Variant
from rdds.lib.vcf import ParsableVariant
from .class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT

# Types definitions
info_meta_field = Dict[str, str]
InfoFormatDict: Dict[str, info_meta_field] = {}

_NUMPY_NUMERICAL_DTYPES = [np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64)]
_MAX_STRING_LENGTH = 2 ** 9
_NUMPY_STRING_DTYPES = [np.dtype('<U%d' % i) for i in range(1, _MAX_STRING_LENGTH + 1)]
_HD5_STRING_DTYPES = [f'|S%d' % i for i in range(1, _MAX_STRING_LENGTH)]


@dataclass
class TaskMetaData:
    input_vcf_file_path: str  # Path of vcf to be parsed
    field_name_to_parse: str  # The actual field name to parse
    parsed_field_name_regexp_fmt: str  # Regexp match pattern for parsed variants to collect data for
    output_vcf_path: str  # Intermediate VCF prefix name for storing on disk before assembly
    result_queue: Queue  # Result queue for ParsedVCFFieldTaskResult


@dataclass
class ParsedVCFFieldResult:
    field_name: str
    data: np.ndarray


@dataclass
class ParsedVCFFieldTaskResult:
    parent_field_name: str  # Name of field that was parsed
    part_hd5_file_path: str  # File path to HD5 file where parsed data is stored


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
        self.out_file_path: str = file_path

    @staticmethod
    def resize_dataset(hd5_group: Hd5Group,
                       increment_size: int) -> Hd5Group:
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
    def add_subset(dataset: Hd5Group,
                   name: str,
                   size: int,
                   numpy_dtype: Type = None,
                   hd5_dtype: Type = None,
                   hd5_fillvalue: Any = None):
        """
        Add a sub dataset to GROUP.
        :param dataset: Dataset group
        :param name: Name of dataset (i.e. the key)
        :param size: Size of allocated dataset
        :param numpy_dtype: Numpy Data type to be translated into hd5 compatible str
        :param hd5_dtype: HD5 file native dtype
        :param hd5_fillvalue: HD5 file native fillvalue
        :return: dataset for chaining
        """
        _dtype = None
        if numpy_dtype is not None:
            if numpy_dtype in _NUMPY_STRING_DTYPES:
                _dtype = (string_dtype('utf-8', _MAX_STRING_LENGTH))
                fillvalue = b'\0'  # String termination
            elif numpy_dtype in _NUMPY_NUMERICAL_DTYPES:
                _dtype = np.float32
                fillvalue = np.nan  # NaN
            else:
                raise ValueError(f'No Numpy -> HD5PY datatype mapping available for Type {numpy_dtype}')
        elif hd5_dtype is not None:
            _dtype = hd5_dtype  # Dtype is already
            fillvalue = hd5_fillvalue
        else:
            raise ValueError('No dtype is provided')
        dataset.create_dataset(name=name,
                               shape=(size, 1),
                               maxshape=(None, 1),
                               dtype=_dtype,
                               fillvalue=fillvalue)
        return dataset

    @staticmethod
    def _parse_collect_vcf_field(task_meta_data: TaskMetaData):
        """
        Multiprocessing capable method that performs parsing and collection of VCF data.
        Parse data row by row, so there's a 1:1 mapping between row in VCF and row in hd5 file.
        :param task_meta_data: Instance processing arguments
        :return: ParsedVCFFieldResult
        """
        vcf_reader: VCFReader = VCFReader(task_meta_data.input_vcf_file_path, 'r', lazy=True)
        n_variants: float = float(vcf_reader.number_of_variants)
        # Create output hd5 file where data is stored
        tmpfile: NamedTemporaryFile = NamedTemporaryFile(dir=os.path.dirname(task_meta_data.output_vcf_path),
                                                         prefix='tmp-'+task_meta_data.field_name_to_parse+'-',
                                                         suffix='.hd5',
                                                         delete=False)
        hd5_out_file_path: str = tmpfile.name
        hd5_file = Hd5File(tmpfile, 'w')
        chunk_size = 10000

        if n_variants <= chunk_size:
            chunk_size = int(n_variants)

        def get_new_chunk_list() -> List[str]:
            return ['' for _ in range(0, chunk_size)]

        chunk: Dict[str, Any] = dict()
        chunk_idx: int = 0
        last_progress_update_value: float = 0
        for idx, unparsed_variant in enumerate(vcf_reader):
            progress_percent: float = 100 * (float(idx) / n_variants)
            if progress_percent - last_progress_update_value > 5.0:
                MULTIPROCESSING_LOGGER.info(f'{progress_percent:.1f}%')
                last_progress_update_value = progress_percent
            unparsed_variant: Variant = unparsed_variant
            parsed_variant: ParsableVariant = ParsableVariant(unparsed_variant,
                                                              parse_only_fields=[task_meta_data.field_name_to_parse],
                                                              vep_csq_description=vcf_reader.csq_description)
            # ParsedVariant unpacks nested fields into separate attributes, so loop over the children
            # attributes to 'field_name_to_parse'. Match patterns that begin with FIELD_NAME_TO_PARSE.
            # FIXME: Mapping in ParsableVariant from parent -> children name to avoid name clash?s
            matching_parsed_attributes: List[Any] = [match(task_meta_data.parsed_field_name_regexp_fmt, attribute)
                                                     for attribute in parsed_variant.parsed_fields]
            # Drop Nones
            matching_parsed_attributes: List[Match] = \
                [entry for entry in matching_parsed_attributes if entry is not None]
            matching_parsed_attributes: List[str] = \
                [entry.string for entry in matching_parsed_attributes]  # Get actual name: string
            # Store values to chunk dict
            for parsed_attribute in matching_parsed_attributes:
                if parsed_attribute not in chunk.keys():
                    chunk.update({parsed_attribute: get_new_chunk_list()})
                chunk[parsed_attribute][chunk_idx] = parsed_variant.get_attribute(parsed_attribute)
            chunk_idx += 1
            # Store to disk if chunk is full or current variant is the last one in VCF file
            if chunk_idx >= chunk_size or idx >= n_variants - 1:
                # Set start stop indexes based on: full chunk, almost full chunk
                hd5_chunk_start_idx: int = idx - chunk_idx + 1  # Adjust for prior chunk_idx addition, subtract 1
                hd5_chunk_stop_idx: int = idx + 1  # Exclusive indexing in python on upper bound
                for chunk_attribute_name, chunk_list in chunk.items():
                    if chunk_attribute_name not in list(hd5_file.keys()):
                        # Assume initially all data types are floats
                        hd5_file.create_dataset(name=chunk_attribute_name,
                                                shape=(n_variants,),
                                                dtype=np.float32,  # Tensorflow GPU only supports 32bits precision
                                                fillvalue=np.nan)
                    try:
                        if hd5_file[chunk_attribute_name].dtype == np.float32:
                            # Try to preserve float dtype if field is empty
                            chunk_float_copy = \
                                [np.nan if isinstance(chunk_value, str) and chunk_value == '' else chunk_value for chunk_value in chunk_list]
                            chunk_list_modified = [float(chunk_value) for chunk_value in chunk_float_copy]
                        elif hd5_file[chunk_attribute_name].dtype == h5py.string_dtype():
                            chunk_list_modified = [str(chunk_value) for chunk_value in chunk_list]
                        else:
                            # FIXME: Value error exception raise and try catch wrapping this
                            raise NotImplementedError(f'Unknown dataset dtype {hd5_file[chunk_attribute_name].dtype}')
                        hd5_file[chunk_attribute_name][hd5_chunk_start_idx: hd5_chunk_stop_idx] = \
                            chunk_list_modified[0: chunk_idx]
                    except (TypeError, ValueError) as error:
                        MULTIPROCESSING_LOGGER.debug(
                            f'{chunk_attribute_name} is string type from here on after: {error} due to {set(chunk_list)}')
                        hd5_file.move(chunk_attribute_name, chunk_attribute_name+'-float')
                        # If storage as floats fails, change dataset dtype to string (actually bytes)
                        # String data is stored as bytes, variable length
                        hd5_file.create_dataset(name=chunk_attribute_name,
                                                shape=(n_variants,),
                                                dtype=h5py.string_dtype(),
                                                fillvalue=b'\0')
                        # Transfer existing data in float dataset to str type dataset
                        MULTIPROCESSING_LOGGER.debug(f'Transferring existing data ({idx} samples)')
                        float_data = hd5_file[chunk_attribute_name + '-float'][0:idx]
                        # FIXME: only 1d support
                        # TODO: Improve speed by chunking read here
                        if len(float_data.shape) > 1:
                            raise NotImplementedError(f'Only 1D supported, got {float_data.shape}')
                        hd5_file[chunk_attribute_name][0:idx] = [str(value) if value == value else b'' for value in float_data.astype(list)]
                        del float_data
                        # Store failed chunk as string
                        hd5_file[chunk_attribute_name][hd5_chunk_start_idx: hd5_chunk_stop_idx] = \
                            [str(chunk_value) for chunk_value in chunk_list][0: chunk_idx]
                        # Drop the old dataset
                        del hd5_file[chunk_attribute_name + '-float']
                        hd5_file.flush()
                        gc.collect()
                chunk = dict()
                chunk_idx = 0
            del unparsed_variant, parsed_variant, matching_parsed_attributes
        vcf_reader.close()
        hd5_file.flush()
        hd5_file.close()
        tmpfile.close()
        del vcf_reader
        del hd5_file
        del tmpfile
        gc.collect()
        os.sync()

        parsed_vcf_field_result: ParsedVCFFieldTaskResult = \
            ParsedVCFFieldTaskResult(parent_field_name=task_meta_data.field_name_to_parse,
                                     part_hd5_file_path=hd5_out_file_path)
        MULTIPROCESSING_LOGGER.debug(f'[{task_meta_data.field_name_to_parse}] Submitting result')
        # FIXME: Put fails if DummyPool because it's closed by previous job
        task_meta_data.result_queue.put(parsed_vcf_field_result)
        MULTIPROCESSING_LOGGER.info(f'[{task_meta_data.field_name_to_parse}] Completed. {ProcessResourceUsage()}')
        task_meta_data.result_queue.close()
        task_meta_data.result_queue.join_thread()

    def _parse_vcf_fields_to_hd5s(self,
                                  vcf_file: str,
                                  parsed_vcf_intermediate_storage_path: str):
        """
        For every field in VCF, create a HD5 file with the data to be assembled later.
        Processing is run asynchronously.

        :param vcf_file:
        :return: multiprocessing results iterable async variant, pool handle
        """

        vcf_reader: VCFReader = VCFReader(vcf_file, 'r', lazy=False)
        MULTIPROCESSING_LOGGER.info(f'Parsing: {vcf_file}[{vcf_reader.data_fields}]')

        # Find all fields in VCF that can be parsed. Parse CSQ sub fields separately because this field is so deep
        tasks = []
        for field_name in vcf_reader.data_fields:
            if field_name == 'CSQ':
                for csq_sub_field_name in vcf_reader.csq_sub_fields:
                    tasks.append(TaskMetaData(input_vcf_file_path=vcf_file,
                                              field_name_to_parse='CSQ',
                                              parsed_field_name_regexp_fmt=f'{re.escape(csq_sub_field_name)}.*',
                                              output_vcf_path=f'{parsed_vcf_intermediate_storage_path}',
                                              result_queue=ProcessPool.get_context().Queue()))
            else:
                tasks.append(TaskMetaData(input_vcf_file_path=vcf_file,
                                          field_name_to_parse=field_name,
                                          parsed_field_name_regexp_fmt=f'{re.escape(field_name)}.*',
                                          output_vcf_path=f'{parsed_vcf_intermediate_storage_path}',
                                          result_queue=ProcessPool.get_context().Queue()))
        if len(tasks) == 0:
            raise ValueError(f'No VCF fields to parse in {vcf_file}')
        vcf_reader.close()

        # Run multi core processing of VCF file
        # Assign every process a separate result_queue to avoid pipe fill-up locking up all processes.
        process_pool: ProcessPool = ProcessPool(fn=self._parse_collect_vcf_field,
                                                args=[(task, ) for task in tasks],  # Pool expects tuple arguments per process
                                                process_names=[task.parsed_field_name_regexp_fmt for task in tasks])

        hd5_out_file: Hd5File = Hd5File(name=self.out_file_path, mode='w')
        # Setup a dataset with dimensions [n_variants, feature_dimension]
        structured_vcfs: Hd5Group = hd5_out_file.create_group('structured_vcfs')

        completed_task_queue: CompletedTaskQueue = process_pool.run_async()

        # Assemble all VCFs to a single hd5 file, flattened
        for result_idx in range(0, process_pool.n_expected_tasks):
            completed_task: Task = completed_task_queue.get()
            MULTIPROCESSING_LOGGER.info(f'Retrieved {completed_task.process.name}')
            MULTIPROCESSING_LOGGER.debug(f'[MAIN]{ProcessResourceUsage()}')
            if not completed_task.process.exitcode == 0:
                raise RuntimeError(f'Processing failed: {completed_task}')
            task_args: TaskMetaData = completed_task.args[0]
            result: ParsedVCFFieldTaskResult = task_args.result_queue.get(timeout=1)  # Fail if data is missing
            task_args.result_queue.close()
            hd5_part: Hd5File = Hd5File(result.part_hd5_file_path, 'r')
            MULTIPROCESSING_LOGGER.info(f'{result.part_hd5_file_path} content: {list(hd5_part.items())}')
            for part_dataset_name, part_data_set in hd5_part.items():
                if part_dataset_name in set(structured_vcfs.keys()):
                    # TODO: Don't parse duplicate datasets. This is due to regexp and dynamic naming of fields in ParsableVariant
                    MULTIPROCESSING_LOGGER.warning(f'Won\'t add duplicate dataset \'{part_dataset_name}\': {completed_task}')
                    continue
                MULTIPROCESSING_LOGGER.info(f'Adding {part_dataset_name} {part_dataset_name}')
                structured_vcfs.create_dataset(name=part_dataset_name,
                                               shape=part_data_set.shape,
                                               dtype=part_data_set.dtype,
                                               fillvalue=part_data_set.fillvalue)
                structured_vcfs[part_dataset_name][()] = part_data_set[()]  # Copy all data from part file to main file
                hd5_out_file.flush()
                MULTIPROCESSING_LOGGER.info(f'Added dataset {part_dataset_name}')
            hd5_part.close()
            remove(result.part_hd5_file_path)
        hd5_out_file.close()
        del hd5_out_file
        process_pool.close()
        del process_pool
        MULTIPROCESSING_LOGGER.info(f'[MAIN]{ProcessResourceUsage()}')

    def compile_structured_format(self,
                                  vcf_file: str,
                                  features: List[str] = None,
                                  features_ignore: List[str] = None):
        """
        Args:
            vcf_files: List of variant files to be used for dataset creation
            features: The INFO/FORMAT fields to add to dataset (matching VCF field names, e.g. CLINVAR_CLNSIG)
            features_ignore: List of INFO/FORMAT fields to ignore
        """

        time_start = datetime.now()

        # Default ignored features
        if features_ignore is None:
            features_ignore = []

        # Drop ignore features
        for drop_feature in features_ignore:
            try:
                features.remove(drop_feature)
            except KeyError:
                pass

        # TODO: Support for selecting subset features

        self._parse_vcf_fields_to_hd5s(vcf_file=vcf_file,
                                       parsed_vcf_intermediate_storage_path=f'{self.out_file_path}.')

        print(f'Dataset {self.out_file_path} generation complete')
        print(f'Took {datetime.now() - time_start}')

    def compile_structured_format_mutacc_tp_cases(self, mutacc_vcf_file_path: str):
        """
        Add a truth label in column 'label' as 0: TN/ unconfirmed, 1: TP, confirmed pathogenic variant.

        If there's a matching variant in this (previously compiled) dataset, consider this
        variant also duplicated in MUTACC VCF file as pathogenic by setting label == 1.0.

        Variants in this datafile (self.file_path) that's not duplicated in MUTACC will be considered TN.

        :param mutacc_vcf_file_path: File path to mutacc VCF dump file.
        (All variants in this file is considered TP pathogenic variants).
        """
        with Hd5File(self.file_path, 'r+') as dset_file:
            # Since the amount of TP cases are small it's affordable to keep them in RAM for processing speed
            tp_variants: List[ParsableVariant] = []
            vcf_reader: VCFReader = VCFReader(mutacc_vcf_file_path)
            for variant in vcf_reader:
                tp_variants.append(ParsableVariant(variant))
            structured_vcfs = dset_file['structured_vcfs']
            dlen = structured_vcfs['ID'].shape[0]  # outermost dimension is n_variants
            structured_vcfs.create_dataset(name='label',
                                           shape=(dlen, 1),
                                           maxshape=(dlen, 1),
                                           dtype=np.float32,
                                           fillvalue=LABEL_BENIGN_VARIANT)
            for idx in range(0, dlen):
                # To be able to compare variant attributes, datatypes must be identical (python standard dtypes)
                chrom = (structured_vcfs['CHROM'][idx, 0]).decode('utf-8')
                pos = int(structured_vcfs['pos'][idx, 0])
                ref = (structured_vcfs['ref'][idx, 0]).decode('utf-8')
                alt = (structured_vcfs['alt'][idx, 0]).decode('utf-8')
                # For every variant in MUTACC set, mark variant as causative if found in existing dataset
                for disease_causing_variant in tp_variants:
                    if disease_causing_variant.CHROM == chrom and \
                       disease_causing_variant.POS == pos and \
                       disease_causing_variant.REF == ref and \
                       disease_causing_variant.ALT == alt:
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
                vcf_file: str,
                features: List[str] = None,
                vcf_file_mutacc_confirmed_tp_cases: str = None):
        self.compile_structured_format(vcf_file=vcf_file,
                                       features=features)
        if vcf_file_mutacc_confirmed_tp_cases is not None:
            self.compile_structured_format_mutacc_tp_cases(mutacc_vcf_file_path=vcf_file_mutacc_confirmed_tp_cases)

    def view(self):
        """
        Print dataset meta info to stdout such as groups, shapes and dtypes.
        :return:
        """

        def print_dataset(name: str, group: Hd5Group):
            s = f'{self.out_file_path}::{name}: '
            def add_attrib(attrib_name):
                nonlocal s
                try:
                    s += f'{attrib_name}={group.__getattribute__(attrib_name)} '
                except AttributeError:
                    pass
            add_attrib('shape')
            add_attrib('dtype')
            print(s)

        with Hd5File(self.out_file_path, 'r') as h5py_file:
            h5py_file.visititems(print_dataset)
