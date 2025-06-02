import progressbar.widgets
from progressbar import ProgressBar
import numpy as np
import pandas as pd
import gc
import seaborn as sb
import matplotlib.pyplot as plt
from typing import List
import os
from cyvcf2 import Writer
import json

from rdds.lib.vcf import VCFReader, ParsableVariant
from rdds.lib.logging import get_logger
from .. import WORKDIR

FIGSIZE = (30, 20)
_LOGGER = get_logger('vcf-rank-results', log_level='debug')


def _get_pathogenic_variant_ids(vcf_file_path: str) -> List[str]:
    """
    Unfortunately, the exported [case_name]_pathogenic.vcf _may_ not contain a VCF
    header. So this file has to be parsed manually for the variant IDs.

    Return pathogenic variant IDs
    :param vcf_file_path: Input file containing pathogenic variants
    :return: List of VCF IDs
    """

    ids: List[str] = []
    with open(vcf_file_path, 'r') as file:
        for line in file:
            # If there's a header, skip it
            if line.startswith('#'):
                continue
            # First variant in VCF
            parts = line.split('\t')  # split on tab character
            variant_id = parts[2]  # ID field is the 3:rd field in a standard VCF
            ids.append(variant_id)
    return ids


def view_vcf_rank_results(vcf_file_path: str,
                          vcf_pathogenic_path: str,
                          workdir: str):
    """
    View ranking results on VCF
    :param vcf_file_path: Input VCF path
    :param vcf_pathogenic_path: Path to pathogenic VCF file (true positive variant)
    :param workdir: Directory where to store output files
    :return:

    Viewing result files:
    cat vrs-highest-ranked.csv | column -t --separator ,| less

    """
    if not os.path.exists(workdir):
        _LOGGER.debug(f'Creating workdir {workdir}')
        os.mkdir(workdir)

    _LOGGER.debug(f'Loading VCF {vcf_file_path}')
    vcf_reader = VCFReader(fname=vcf_file_path)
    variants = list(vcf_reader)  # Load variants to RAM
    n_variants = len(variants)
    pos = np.empty(n_variants)
    chrom = np.empty(n_variants, dtype=object)
    ref = np.empty(n_variants, dtype=object)
    alt = np.empty(n_variants, dtype=object)
    id = np.empty(n_variants, dtype=object)
    genmod_rank_score = np.empty(n_variants)
    genmod_rank_score_normalized = np.empty(n_variants)
    vrs_rank_score = np.empty(n_variants)
    frq = np.empty(n_variants)
    vrs_model_explanation = np.empty(n_variants, dtype=object)
    parse_only_fields = ['POS', 'CHROM', 'RankScore', 'RankScoreNormalized', 'VrsModelPrediction', 'Frq']
    pbar = ProgressBar(max_value=n_variants)
    _LOGGER.info(f'{vcf_file_path}, n={n_variants} variants')
    for i, variant in enumerate(variants):
        parsed_variant = ParsableVariant(variant=variant,
                                         parse_only_fields=parse_only_fields,
                                         vep_csq_description=vcf_reader.csq_description)
        try:
            frq[i] = variant.INFO['Frq']
        except KeyError:
            frq[i] = np.nan
        chrom[i] = parsed_variant.CHROM
        pos[i] = parsed_variant.POS
        ref[i] = parsed_variant.REF
        alt[i] = parsed_variant.ALT
        id[i] = parsed_variant.ID
        if 'RankScore_value' in parsed_variant.parsed_fields:
            genmod_rank_score[i] = parsed_variant.RankScore_value
        else:
            genmod_rank_score[i] = np.nan
        if 'RankScoreNormalized_value' in parsed_variant.parsed_fields:
            genmod_rank_score_normalized[i] = parsed_variant.RankScoreNormalized_value
        else:
            genmod_rank_score_normalized[i] = np.nan
        if 'VrsModelPrediction' in parsed_variant.parsed_fields:
            vrs_rank_score[i] = parsed_variant.VrsModelPrediction
        else:
            vrs_rank_score[i] = np.nan
        try:
            vrs_model_explanation[i] = variant.INFO['VrsModelExplanation']
        except KeyError:
            pass
        pbar.update(i)
    pbar.finish()
    vcf_reader.close()
    del variants
    gc.collect()

    df = pd.DataFrame(data=
                      {
                          'chrom': chrom,
                          'pos': pos,
                          'ref': ref,
                          'alt': alt,
                          'id': id,
                          'frq': frq,
                          'genmod_rank_score': genmod_rank_score,
                          'genmod_rank_score_normalized': genmod_rank_score_normalized,
                          'vrs_rank_score': vrs_rank_score,
                          'vrs_model_explanation': vrs_model_explanation
                      })

    df['residuals'] = df.genmod_rank_score_normalized - df.vrs_rank_score

    # Make sure unseed samples are considered RAREST, not to discard these samples
    # NOTE: This modifies data in the CSV files.
    df['frq'] = df.frq.fillna(0.0)
    # Frequency filter the VRS model predictions
    df_frqfilt = df[df.frq <= 1.0/2000.0].copy()

    pathogenic_variant_ids = _get_pathogenic_variant_ids(vcf_file_path=vcf_pathogenic_path)
    if len(pathogenic_variant_ids) == 0:
        raise ValueError(f'Expected some pathogenic variants, but found none in {vcf_pathogenic_path}')
    pathogenic_variants = df.query(f'id == {pathogenic_variant_ids}')
    """
    In case of pre-applied clinical gene panel filter, pathogenic variants might be lost from the inference VCF.
    If so, make note of this and don't try to compute performance scores, plots etc since there's no
    data to visualize.
    """
    available_inferred_pathogenic_variants = True
    if len(pathogenic_variants) == 0:  # Check for pathogenic variant(s) in inference VCF.
        available_inferred_pathogenic_variants = False
    if available_inferred_pathogenic_variants and len(pathogenic_variants) != len(pathogenic_variant_ids):
        raise ValueError(f'Expected identical amounts of pathogenic variants,\
got {pathogenic_variant_ids}!={pathogenic_variants}')

    def _write_performance_summary(performance_summary: dict):
        with open(os.path.join(workdir, 'performance.json'), 'w') as file:
            meta_str = json.dumps(performance_summary, indent=2)
            file.write(meta_str)

    # Initialize empty summary performance statistic in case of not available_inferred_pathogenic_variants
    performance_summary = {
        'pathogenic_variants': pd.DataFrame().to_dict(),
        'genmod_rank': pd.DataFrame().to_dict(),
        'vrs_rank': pd.DataFrame().to_dict(),
        'vrs_rank_frqfilt': pd.DataFrame().to_dict(),
        'available_inferred_pathogenic_variants': available_inferred_pathogenic_variants
    }

    if not available_inferred_pathogenic_variants:
        _write_performance_summary(performance_summary)
        _LOGGER.info(f'Completed analysis of case {vcf_file_path}, {vcf_pathogenic_path} (no pathogenic variant(s) in VCF)')
        return

    fig: plt.Figure = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    sb.violinplot(df[['vrs_rank_score', 'genmod_rank_score_normalized']], ax=ax)
    fig.savefig(os.path.join(workdir, 'violin.png'))

    fig: plt.Figure = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    sb.boxplot(df[['vrs_rank_score', 'genmod_rank_score_normalized']], ax=ax)
    fig.savefig(os.path.join(workdir, 'box.png'))

    fig: plt.Figure = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    sb.scatterplot(df[['vrs_rank_score', 'genmod_rank_score']],
                   x='vrs_rank_score',
                   y='genmod_rank_score',
                   ax=ax,
                   alpha=0.5,
                   marker='.')
    sb.scatterplot(pathogenic_variants[['vrs_rank_score', 'genmod_rank_score']],
                   x='vrs_rank_score',
                   y='genmod_rank_score',
                   ax=ax,
                   color='red',
                   marker='o')
    fig.savefig(os.path.join(workdir, 'scatter.png'))

    fig: plt.Figure = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    sb.scatterplot(df[['vrs_rank_score', 'frq']],
                   x='vrs_rank_score',
                   y='frq',
                   ax=ax,
                   alpha=0.5,
                   marker='.')
    sb.scatterplot(pathogenic_variants[['vrs_rank_score', 'frq']],
                   x='vrs_rank_score',
                   y='frq',
                   ax=ax,
                   color='red',
                   marker='o')
    fig.savefig(os.path.join(workdir, 'scatter-frq-rank.png'))

    fig: plt.Figure = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    sb.scatterplot(df_frqfilt[['vrs_rank_score', 'frq']],
                   x='vrs_rank_score',
                   y='frq',
                   ax=ax,
                   alpha=0.5,
                   marker='.')
    sb.scatterplot(pathogenic_variants[['vrs_rank_score', 'frq']],
                   x='vrs_rank_score',
                   y='frq',
                   ax=ax,
                   color='red',
                   marker='o')
    fig.savefig(os.path.join(workdir, 'scatter-frq-rank-frqfilted.png'))

    _LOGGER.debug('Plot generation complete')

    # Save to file for later inspection
    store_n_variants = 500

    def _sort_by_score_save_csv_get_rank(df: pd.DataFrame,
                                         sort_column_name: str,
                                         file_name: str,
                                         top_n_variants: int,
                                         additional_output_columns: List[str] = None) -> pd.DataFrame:
        with open(os.path.join(workdir, file_name), 'w') as file:
            df_sorted = df.sort_values(sort_column_name, ascending=False).copy()
            df_sorted.reset_index(inplace=True)
            df_sorted.iloc[0:top_n_variants].to_csv(file)
            _LOGGER.debug(f'Stored {file}')
            output_columns = ['id', sort_column_name]
            if additional_output_columns:
                output_columns.extend(additional_output_columns)
            return df_sorted.query(f'id == {pathogenic_variant_ids}')[output_columns]

    ranks_vrs = _sort_by_score_save_csv_get_rank(df=df,
                                                 sort_column_name='vrs_rank_score',
                                                 file_name='vrs-highest-ranked.csv',
                                                 top_n_variants=store_n_variants,
                                                 additional_output_columns=['vrs_model_explanation'])
    ranks_vrs_frqfilt = _sort_by_score_save_csv_get_rank(df=df_frqfilt,
                                                         sort_column_name='vrs_rank_score',
                                                         file_name='vrs-highest-ranked-frqfilt.csv',
                                                         top_n_variants=store_n_variants,
                                                         additional_output_columns=['vrs_model_explanation'])
    ranks_genmod = _sort_by_score_save_csv_get_rank(df=df,
                                                    sort_column_name='genmod_rank_score',
                                                    file_name='genmod-highest-ranked.csv',
                                                    top_n_variants=store_n_variants)

    _LOGGER.debug('CSV storage complete')

    def _write_variants_to_vcf(indexes: np.ndarray,
                               file_name: str):
        """
        Copy variants from input VCF to a new VCF as a subset based on indexes
        """
        pbar = ProgressBar(widgets=[progressbar.widgets.BouncingBar()],
                           prefix=file_name)
        pbar.start()
        indexes = list(indexes)
        vcf_reader = VCFReader(fname=vcf_file_path)
        output_file_name = os.path.join(workdir, file_name)
        vcf_writer: Writer = Writer(output_file_name, vcf_reader, mode='w')
        variants = list(vcf_reader)  # Load to RAM
        # Write variants in sorted order to output VCF
        for index in indexes:
            vcf_writer.write_record(variants[index])
        vcf_reader.close()
        vcf_writer.close()
        gc.collect()
        pbar.finish()

    _write_variants_to_vcf(indexes=df.sort_values('genmod_rank_score',
                                                  ascending=False).iloc[0:store_n_variants].index.values,
                           file_name='genmod-top-scores.vcf')

    _write_variants_to_vcf(indexes=df.sort_values('vrs_rank_score',
                                                  ascending=False).iloc[0:store_n_variants].index.values,
                           file_name='vrs-top-scores.vcf')

    _write_variants_to_vcf(indexes=df_frqfilt.sort_values('vrs_rank_score',
                                                          ascending=False).iloc[0:store_n_variants].index.values,
                           file_name='vrs-top-scores-frqfilt.vcf')

    _LOGGER.debug('Ranked VCF storage complete')

    # Store annotated, ranked pathogenic variants to separate VCF
    _write_variants_to_vcf(indexes=pathogenic_variants.index.values,
                           file_name='pathogenic_variants.vcf')
    _LOGGER.debug('Pathogenic variants storage complete')

    performance_summary.update({
        'pathogenic_variants': pathogenic_variants.to_dict(),
        'genmod_rank': ranks_genmod.to_dict(),
        'vrs_rank': ranks_vrs.to_dict(),
        'vrs_rank_frqfilt': ranks_vrs_frqfilt.to_dict()
    })

    _write_performance_summary(performance_summary)

    _LOGGER.info(f'Completed analysis of case {vcf_file_path}, {vcf_pathogenic_path}')
