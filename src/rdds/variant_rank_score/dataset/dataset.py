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
from rdds.lib.process_pool import ProcessPool, DummyPool, Task, CompletedTaskQueue, MULTIPROCESSING_LOGGER
from dataclasses import dataclass
from re import match, Match
from os import remove
import re

from rdds.lib.resource_usage import ProcessResourceUsage
from rdds.lib.vcf import VCFReader, Variant
from rdds.lib.vcf import ParsableVariant
from .clinvar_label_mapping import CLINVAR_CLNSIG_DROP_LABELS, CLINVAR_LABEL_MAPPING

# Types definitions
info_meta_field = Dict[str, str]
InfoFormatDict: Dict[str, info_meta_field] = {}

_NUMPY_NUMERICAL_DTYPES = [np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64)]
_MAX_STRING_LENGTH = 2 ** 9
_NUMPY_STRING_DTYPES = [np.dtype('<U%d' % i) for i in range(1, _MAX_STRING_LENGTH + 1)]
_HD5_STRING_DTYPES = [f'|S%d' % i for i in range(1, _MAX_STRING_LENGTH)]

_MASK_KEEP_VALUE: int = 0  # Un-masked rows to be kept
_MASK_DROP_VALUE: int = 1  # Masked rows with this value is to be ignored, dropped


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
                 file_path: str,
                 max_n_workers: int = os.cpu_count()):
        """
        param file_path: Path to VCF file to compile
        param max_n_workers: During processing, use max N concurrent process workers.
        NOTE: cpu_count() returns total count on SLURM node, not amount allowed by SLURM scheduler.
        """
        self.out_file_path: str = file_path
        self._max_n_workers = max_n_workers

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
        process_pool: ProcessPool = ProcessPool(function=self._parse_collect_vcf_field,
                                                args=[(task, ) for task in tasks],  # Pool expects tuple arguments per process
                                                process_names=[task.parsed_field_name_regexp_fmt for task in tasks],
                                                workers=self._max_n_workers)

        hd5_out_file: Hd5File = Hd5File(name=self.out_file_path, mode='w')
        # Setup a dataset with dimensions [n_variants, feature_dimension]
        group_name: str = 'structured_vcfs'
        structured_vcfs: Hd5Group = hd5_out_file.create_group(group_name)

        completed_task_queue: CompletedTaskQueue = process_pool.run_async()

        # Assemble all VCFs to a single hd5 file, flattened
        for result_idx in range(0, process_pool.nr_expected_tasks):
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
        return group_name

    @staticmethod
    def _parse_clinvar_bytestring_to_list(byte_string: bytes) -> List[str]:
        """
        Parse a bytestring of CLINVAR_CLNSIG field to list of strings, all lowercase.

        :param byte_string: A byte encoded string matching CLINVAR CLNSIG field.
          Can be made up by multiple keys, such as 'pathogenic|risk_factor'
        :return: A list of keywords, str
        :raises ValueError: In case no keywords were found
        """

        if not isinstance(byte_string, bytes):
            raise ValueError(f'Expected bytes but got {byte_string}')

        clinvar_clnsig: str = byte_string.decode('utf-8')  # Decode bytestring
        clinvar_clnsig = clinvar_clnsig.lower()  # lowercase all keywords
        keywords: List[str] = re.split("[|\|/|,]", clinvar_clnsig)  # Split on characters | / ,
        keywords = [keyword.rstrip('_').lstrip('_') for keyword in keywords]  # Remove prefixed, trailing '_'
        if len(keywords) == 0:
            raise ValueError(f'Expected a CLNSIG field value, got {byte_string}')
        return keywords

    @staticmethod
    def _clinvar_ground_truth_to_categorical_label(byte_string: bytes) -> float:
        """
        Translates a byte (string) to numerical value to be used as categorical label,
        according to CLINVAR_CLNSIG values.

        Mapping according to: https://www.ncbi.nlm.nih.gov/clinvar/docs/clinsig

        :param byte_string: A byte encoded string matching CLINVAR CLNSIG field.
          Can be made up by multiple keys, such as 'pathogenic|risk_factor'
        :return: Float value in range (0, 1)
        """

        def sum_cap_to_bound(*values: Tuple[float]) -> float:
            """
            Merge multiple CLINSIG keywords to a single categorical label by summation.
            Pathogenic keywords get values (1, +inf) but is capped to (1, 1).
            Benign values are 0.0.
            :param values: Tuple of floats
            :return: float value in range (0, 1)
            """
            values_arr = np.array(values)
            sum = np.sum(values_arr)
            label = np.clip(sum, a_min=0, a_max=1)  # Clip to (0, 1)
            if not isinstance(label, (float, np.float64)):
                raise ValueError('Failed to reduce dimensions')
            return label

        keywords = Dataset._parse_clinvar_bytestring_to_list(byte_string=byte_string)
        categorical_labels: List[float] = [CLINVAR_LABEL_MAPPING[keyword] for keyword in keywords]  # Convert to categorical values
        categorical_labels: Tuple[float] = tuple(categorical_labels)  # Reformat to fit input to sum_cap_to_bound()
        categorical_label = sum_cap_to_bound(categorical_labels)
        return categorical_label

    @staticmethod
    def _postprocess_ground_truth(hd5_file_path: str,
                                  group_name: str,
                                  ground_truth_dataset_name: str,
                                  mapping_fn: Callable):
        """
        Convert values in ground_truth_column_name to numerical values, i.e.
        categorical label.

        Supports only 1D labels, 1:1 mapping.

        :param group_name: The group where the labels are found
        :param ground_truth_dataset_name: The name of the column where the labels are found
        :param mapping_fn: A function that translates bytestring to floating point value.
        """
        hd5_out_file = h5py.File(hd5_file_path, 'r+')
        group: h5py.Group = hd5_out_file[group_name]
        ground_truth: np.ndarray = group[ground_truth_dataset_name][:]
        if len(ground_truth.shape) > 1:
            raise ValueError(f'Only 1D labels are supported. Got {ground_truth.shape}')
        dlen = ground_truth.shape[0]
        categorical_label = np.zeros(dlen)
        categorical_label_dset = group.create_dataset(name='label',
                                                      shape=dlen,
                                                      dtype=np.float32)
        for i in range(0, dlen):
            categorical_label[i] = mapping_fn(ground_truth[i])
        categorical_label_dset[:] = categorical_label
        hd5_out_file.flush()
        hd5_out_file.close()

    def _compile_structured_format(self,
                                   vcf_file: str):
        """
        Compile a VCF into a CSV-like columnar format in HD5 protocol.
        :param vcf_file: The VCF file to be used for dataset creation
        # TODO: Support for selecting subset features
        """
        group_name = self._parse_vcf_fields_to_hd5s(vcf_file=vcf_file,
                                                    parsed_vcf_intermediate_storage_path=f'{self.out_file_path}.')
        return group_name

    @staticmethod
    def _mask_use_only_clinvar_benign_pathogenic(byte_string: bytes) -> int:
        """
        Return MASK or KEEP value depending on contents in byte_string.
        If MASK value, the data row should be ignored in further processing.
        :param byte_string: Clinvar CLNSIG field value, can be multi-type, 'likely_pathogenic|uncertain_significance'
        :return: Integer, representing masking or not.
        """
        clinvar_keywords = Dataset._parse_clinvar_bytestring_to_list(byte_string=byte_string)
        for drop_label in CLINVAR_CLNSIG_DROP_LABELS:
            if drop_label in clinvar_keywords:
                return _MASK_DROP_VALUE
        return _MASK_KEEP_VALUE

    @staticmethod
    def _apply_data_masking_by_label(hd5_file_path: str,
                                     group_name: str,
                                     ground_truth_dataset_name: str,
                                     mask_function: Callable) -> str:
        """
        Drops data samples according to mask_function.

        NOTE: This method requires that all data fits into RAM!

        :param hd5_file_path: Path to the HD5 file to apply masking to
        :param group_name: The group where data resides
        :param mask_function: Function that computes a mask.
        :return: The processed group name containing the masked data.
        :raises ValueError: In case datasets not rank 1
        """
        hd5_out_file = h5py.File(hd5_file_path, 'r+')
        group: h5py.Group = hd5_out_file[group_name]
        ground_truth: np.ndarray = group[ground_truth_dataset_name][:]
        if len(ground_truth.shape) > 1:
            raise ValueError(f'Only 1D labels are supported. Got {ground_truth.shape}')
        data_length: int = ground_truth.shape[0]
        data_mask = np.zeros(data_length)
        for i in range(0, data_length):
            data_mask[i] = mask_function(ground_truth[i])
        idx_to_keep: np.ndarray = np.argwhere(data_mask == _MASK_KEEP_VALUE)[:, 0]  # Remove outer dim
        print(f'Masking reduces amount of data to {100 * (float(len(idx_to_keep)) / data_length)} %')
        masked_group_name: str = f'{group_name}-masked'
        masked_group: h5py.Group = hd5_out_file.create_group(masked_group_name)
        for dataset_name in group.keys():
            if group[dataset_name].ndim > 1:
                raise NotImplementedError(f'Only 1D datasets supported, got {dataset_name}:{group[dataset_name].shape}')
            masked_dataset = masked_group.create_dataset(name=dataset_name,
                                                         shape=(len(idx_to_keep), ),
                                                         dtype=group[dataset_name].dtype)
            data = group[dataset_name][:]  # Load into RAM for performance
            masked_dataset[()] = data[idx_to_keep]
        hd5_out_file.flush()
        hd5_out_file.close()
        return masked_group_name

    @staticmethod
    def _split_to_train_test_sets(hd5_file_path: str,
                                  group_name: str,
                                  ratio_test: float):
        """
        Split group_name into two new groups /train and /test transferring
        all datasets.

        Data samples are randomly selected from the datasets.

        :param hd5_file_path: The HD5 file to work with
        :param group_name: The group to be split (all datasets are transferred)
        :param ratio_test: The ratio of the test dataset [0, 1].
        :return:
        """
        hd5_out_file = h5py.File(hd5_file_path, 'r+')
        group: h5py.Group = hd5_out_file[group_name]

        # Setup array with sample index that will be split into train, test
        dlen = group[list(group.keys())[0]].shape[0]
        sample_idx = np.arange(0, dlen)
        rng: np.random.Generator = np.random.default_rng(seed=0)
        rng.shuffle(sample_idx)
        split_idx: int = int(np.ceil(ratio_test * dlen))
        sample_idxs_test: np.ndarray = np.sort(sample_idx[0:split_idx])
        sample_idxs_train: np.ndarray = np.sort(sample_idx[split_idx:])

        datasets_to_split: List[str] = list(group.keys())
        for group_name, dataset_idx in zip(['train', 'test'], [sample_idxs_train, sample_idxs_test]):
            group_split = hd5_out_file.create_group(name=group_name)
            for dataset_name in datasets_to_split:
                group_split.create_dataset(name=dataset_name,
                                           shape=(len(dataset_idx)),
                                           dtype=group[dataset_name].dtype)
                data = group[dataset_name][:]  # Do slicing on array in RAM for performance
                group_split[dataset_name][:] = data[dataset_idx]
        hd5_out_file.flush()
        hd5_out_file.close()

    def compile(self,
                vcf_file: str):
        time_start = datetime.now()
        group_name = self._compile_structured_format(vcf_file=vcf_file)
        group_name = self._apply_data_masking_by_label(hd5_file_path=self.out_file_path,
                                                       group_name=group_name,
                                                       ground_truth_dataset_name='CLINVAR_GROUND_TRUTH',
                                                       mask_function=self._mask_use_only_clinvar_benign_pathogenic)
        self._postprocess_ground_truth(hd5_file_path=self.out_file_path,
                                       group_name=group_name,
                                       ground_truth_dataset_name='CLINVAR_GROUND_TRUTH',
                                       mapping_fn=self._clinvar_ground_truth_to_categorical_label)
        self._split_to_train_test_sets(hd5_file_path=self.out_file_path,
                                       group_name=group_name,
                                       ratio_test=0.25)
        print(f'Dataset {self.out_file_path} generation complete')
        print(f'Took {datetime.now() - time_start}')

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
