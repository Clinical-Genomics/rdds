import os
from os import PathLike
import pathlib
import gc
from time import time
from typing import List, Set, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd

from rdds.lib.logging import get_logger; _LOGGER = get_logger('vcf2csv', 'info')
from rdds.lib.vcf import VCFReader, Variant, ParsableVariant
from rdds.lib.resource_usage import ProcessResourceUsage
from rdds.lib.workdir import get_workdir_path


class Vcf2Csv:

    def __init__(self,
                 workdir: pathlib.PurePath = pathlib.PurePath(get_workdir_path('vcf-explorer')),
                 figsize=(25, 10)):
        self._workdir: pathlib.PurePath = workdir
        os.makedirs(self._workdir, exist_ok=True)
        self._figsize: Tuple[int, int] = figsize

    @staticmethod
    def _load_variants(vcf_path: PathLike) -> Tuple[List[ParsableVariant], Set[str]]:
        t_start = time()
        _LOGGER.debug(ProcessResourceUsage())
        _LOGGER.info(f'Loading VCF {vcf_path}')
        vcf_reader: VCFReader = VCFReader(vcf_path)
        _LOGGER.debug(f'Contains n variants: {vcf_reader.number_of_variants}')
        variants: List[Variant] = list(vcf_reader)  # Load all into RAM
        vcf_reader.close()
        gc.collect()
        _LOGGER.debug(ProcessResourceUsage())
        _LOGGER.debug(f'Parsing variants')
        parsed_variants = [ParsableVariant(variant=variant,
                                           vep_csq_description=vcf_reader.csq_description) for variant in variants]
        _LOGGER.debug(f'Loading took {time() - t_start}s')
        _LOGGER.info(ProcessResourceUsage())
        del variants
        gc.collect()

        annotations: Set[str] = set()
        for parsed_variant in parsed_variants:
            annotations.update(parsed_variant.parsed_fields)

        return parsed_variants, annotations

    @staticmethod
    def _tabularize(variants: List[ParsableVariant],
                    annotations: Set[str]) -> pd.DataFrame:
        n_variants = len(variants)
        data: Dict[str, Any] = dict()
        for idx, variant in enumerate(variants):
            variant: ParsableVariant
            for annotation in annotations:
                if annotation not in data.keys():
                    _LOGGER.debug(f'Adding annotation data entry: {annotation}')
                    data[annotation] = np.empty(n_variants, dtype='object')
                # Add data from variant, set to np.NaN if datapoint is missing
                data[annotation][idx] = variant.get_attribute(key=annotation, preferred_dtype=np.float64)

        for annotation in annotations:
            # TODO: Cast indexes to INT64, such as POS
            # Try casting to numerical format
            try:
                data[annotation] = data[annotation].astype(np.float64)
                _LOGGER.debug(f'Casted {annotation} to float')
            except ValueError as e:
                _LOGGER.debug(f'{annotation}: {e}')

        df = pd.DataFrame(data=data)
        del variants
        gc.collect()
        return df

    def convert_vcf_to_csv(self,
                           vcf_path: str) -> pathlib.PurePath:
        """
        Convert a VCF to a CSV and store in workdir.

        NOTE: This is a potential lossy method that depends on the implementation in
        ParsableVariant. Only fields listed in ParsableVariant.parsed_fields
        will be copied to the .csv file.

        :param vcf_path: The path to the VCF
        :return: Path to the CSV
        """
        start = time()
        _LOGGER.info(f'Converting {vcf_path} to CSV format')
        variants, annotations = Vcf2Csv._load_variants(vcf_path)
        df = Vcf2Csv._tabularize(variants=variants, annotations=annotations)
        vcf_path = pathlib.PurePath(vcf_path)
        csv_file_path = vcf_path.with_suffix('.csv')
        csv_path = self._store_df_as_csv(df=df,
                                         out_path=csv_file_path)
        duration = time() - start
        _LOGGER.info(f'{vcf_path} conversion took {duration:.1f}s')
        _LOGGER.info(ProcessResourceUsage())
        return csv_path

    def _store_df_as_csv(self,
                         df: pd.DataFrame,
                         out_path: pathlib.PurePath) -> pathlib.PurePath:
        _LOGGER.debug(f'Storing parsed VCF as CSV file at {out_path}')
        df.to_csv(out_path)
        _LOGGER.info(f'Stored parsed VCF as CSV file at {out_path} [{os.path.getsize(out_path) / 1E9:.2f}GB]')
        return out_path
