from h5py import File as Hd5File, string_dtype, Dataset as Hd5DataSet, Group as Hd5Group
import csv
import os
import numpy as np
import subprocess as sp
from progressbar import ProgressBar
from dataclasses import dataclass
from typing import Any
import pandas as pd
from typing import List, Union
import gc
import seaborn as sb
import matplotlib.pyplot as plt
import tempfile
import shutil

from rdds.variant_rank_score.dataset.class_labels import LABEL_PATHOGENIC_VARIANT
from rdds.lib.process_pool import ProcessPool
import rdds.variant_rank_score.inference_exploration.statfns as statfns
from rdds.lib.logging import get_logger
_LOGGER = get_logger(log_level='info')

FIGSIZE = (30, 20)

# FIXME: genmod_scores to contain case_id to allow genmod legacy scores split on case id.

_MAX_STRING_LENGTH = 512
_NUMPY_STRING_DTYPES = [np.dtype('<U%d' % i) for i in range(1, _MAX_STRING_LENGTH + 1)]

@dataclass
class Hd5Spec:
    csv_column_name: str
    hd5_column_name: str
    csv_parse_dtype: Any
    dtype: Any
    fillvalue: Any


_MIVMIR_SCORES_CSV_TO_HD5_SPEC = [
    Hd5Spec('Case', 'case', int, np.int64, np.nan),
    Hd5Spec('VarID', 'id', str, string_dtype(), b'\0'),
    Hd5Spec('Causative', 'causative', float, np.float32, np.nan),
    Hd5Spec('genmod', 'genmod', float, np.float32, np.nan),
    Hd5Spec('mivmir', 'mivmir', float, np.float32, np.nan),
    Hd5Spec('gicam', 'gicam', float, np.float32, np.nan),
    Hd5Spec('INFO', 'info', str, string_dtype(), b'\0'),
    Hd5Spec('GNOMADAF', 'gnomadaf', float, np.float32, np.nan),
    Hd5Spec('GNOMADAF_popmax', 'gnomadaf_popmax', float, np.float32, np.nan),
    Hd5Spec('Frq', 'frq', float, np.float32, np.nan),
]
_OUTERMOST_MIVMIR_SCORES_HD5_SPEC = _MIVMIR_SCORES_CSV_TO_HD5_SPEC[0:6]  # Not nested fields

_LEGACY_GENMOD_CSV_SPEC = [
    Hd5Spec('ID', 'id', 'str', string_dtype(), b'\0'),
    Hd5Spec('genmodscore', 'genmod', float, np.float32, np.nan)
]

FRQ_FIELDS = ['gnomadaf', 'gnomadaf_popmax', 'frq']  # Parsed variant population frequency fields


def _count_variants_in_csv_file(csv_file_path) -> int:
    r = sp.run(["wc", "-l", csv_file_path], capture_output=True, check=True)
    stdout: str = r.stdout.decode('utf-8')
    n_variants = int(stdout.split(' ')[0]) - 1  # Subtract 1 for CSV header
    return n_variants


def parse_mivmir_genmod_file_with_info_csv(mivmir_scores_csv: str) -> dict:
    """
    Custom CSV parsing method since INFO fields contains CSV separators, that breaks parsing.

    Parses according to _MIVMIR_SCORES_CSV_TO_HD5_SPEC and in addition pop frequency fields.
    """
    sep = ','  # CSV separator
    info_sep = ';'  # INFO field separator
    info_key_sep = '='  # Token separating key name and value
    file = open(mivmir_scores_csv, 'r')
    column_names = file.readline()  # Discard first row
    while True:
        line = file.readline()
        if len(line) == 0:
            break
        parsed_line = {}
        for key in FRQ_FIELDS:
            parsed_line.update({key: np.nan})  # Fallback in case of no data for variant
        for key_idx, (hd5_spec, part) in enumerate(zip(_OUTERMOST_MIVMIR_SCORES_HD5_SPEC, line.split(sep))):
            # Since INFO column contains ',' separator tokens, standard CSV parsing fails, do it manually.
            # Piecing INFO field together is also necessary to allow parsing of later fields.
            if hd5_spec.hd5_column_name == 'info':
                # Stitch together the INFO field
                info_field = ''.join(line.split(sep)[key_idx:]).lower()
                for info_part in info_field.split(info_sep):
                    try:
                        k, v = info_part.split(info_key_sep)
                    except ValueError:
                        continue
                    if k in FRQ_FIELDS:
                        parsed_line.update({k: float(v)})
            else:
                parsed_line.update({hd5_spec.hd5_column_name: hd5_spec.csv_parse_dtype(part)})
        yield parsed_line


def yield_batches(batch_size: int, *args, **kwargs):
    """
    Assemble minibatch of data in RAM for performance.
    """
    samples = {}
    n_samples = 0
    for d in parse_mivmir_genmod_file_with_info_csv(*args, **kwargs):
        for key, value in d.items():
            if key not in samples.keys():
                samples.update({key: [value]})
            else:
                samples[key].append(value)
        n_samples += 1
        if n_samples >= batch_size:
            n_samples = 0
            yield pd.DataFrame(samples)
            samples = {}
    if n_samples > 0:
        yield pd.DataFrame(samples)


def convert_csv_to_hd5(mivmir_scores_csv: str,
                       default_genmod_csv: str,
                       output_file_path: str) -> str:
    _LOGGER.info("Building dataset ...")
    _LOGGER.info(f"mivmir_scores_csv {mivmir_scores_csv}")
    _LOGGER.info(f"default_genmod_csv {default_genmod_csv}")

    n_variants_mivmir = _count_variants_in_csv_file(csv_file_path=mivmir_scores_csv)
    n_variants_genmod = _count_variants_in_csv_file(csv_file_path=default_genmod_csv)
    assert n_variants_mivmir == n_variants_genmod, (n_variants_mivmir, n_variants_genmod)
    n_variants = n_variants_mivmir
    _LOGGER.info(f"Number of variants: {n_variants}")

    try:
        os.remove(output_file_path)
    except FileNotFoundError:
        pass

    batch_size = int(1E4)  # Amount of samples to read from CSV files (for performance)

    hd5_file = Hd5File(output_file_path, 'w')

    # Store MIVMIR and Genmod minimal config (inheritance scores)
    gicam_group: Hd5Group = hd5_file.create_group(name='gicamdata')
    for hd5_spec in _MIVMIR_SCORES_CSV_TO_HD5_SPEC:
        gicam_group.create_dataset(name=hd5_spec.hd5_column_name,
                                   shape=(n_variants, ),
                                   dtype=hd5_spec.dtype,
                                   fillvalue=hd5_spec.fillvalue)
    pbar = ProgressBar(max_value=n_variants / batch_size)
    pbar.start()
    for batch_idx, batch in enumerate(yield_batches(batch_size=batch_size, mivmir_scores_csv=mivmir_scores_csv)):
        pbar.update(batch_idx)
        actual_batch_size = len(batch)  # Adjust for EOF
        start = batch_idx * batch_size
        stop = start + actual_batch_size
        for column in batch.columns:
            gicam_group[column][start:stop] = batch[column].values[()]
    pbar.finish()
    hd5_file.flush()

    # Store legacy Genmod scores
    _LOGGER.info(f"Parsing Genmod legacy scores")
    legacy_genmod_group: Hd5Group = hd5_file.create_group(name='legacy_genmod')
    for hd5_spec in _LEGACY_GENMOD_CSV_SPEC:
        legacy_genmod_group.create_dataset(name=hd5_spec.hd5_column_name,
                                   shape=(n_variants,),
                                   dtype=hd5_spec.dtype,
                                   fillvalue=hd5_spec.fillvalue)
    dtypes = {}
    for hd5_spec in _LEGACY_GENMOD_CSV_SPEC:
        dtypes.update({hd5_spec.csv_column_name: hd5_spec.csv_parse_dtype})
    df = pd.read_csv(filepath_or_buffer=default_genmod_csv,
                     header=0,
                     names=[hd5_spec.hd5_column_name for hd5_spec in _LEGACY_GENMOD_CSV_SPEC],
                     index_col=False,
                     usecols=[2, 5],  # Just load 2, 5 (ID and score)
                     dtype=dtypes,
                     chunksize=batch_size)
    pbar = ProgressBar(max_value=n_variants / batch_size)
    pbar.start()
    for batch_idx, batch in enumerate(df):
        pbar.update(batch_idx)
        actual_batch_size = len(batch)  # Adjust for EOF
        start = batch_idx * batch_size
        stop = start + actual_batch_size
        for column in batch.columns:
            legacy_genmod_group[column][start:stop] = batch[column].values[()]
    pbar.finish()
    hd5_file.flush()
    hd5_file.close()

    _LOGGER.info(f"Completed: {output_file_path}")
    return output_file_path

def find_causative_genmod_variants(hd5_file_path: str):
    """
    Create a new dataset for Genmod legacy data, by adding
    a 'causative' dataset using ground truth from mivmir dataset.

    Do this by looking up variant IDs from MIVMIR marked as
    causative, and set them to 1.0 in genmod legacy dataset.
    """
    # TODO: Use np.nonzero instead of argwhere
    _LOGGER.info("Searching for causative genmod variants in legacy genmod dataset ...")
    hd5_file = Hd5File(hd5_file_path, 'r+')
    labels = hd5_file['gicamdata/causative'][()]
    pathogenic_idx = np.argwhere(labels == 1)[:, 0]
    causative_ids = hd5_file['gicamdata/id'][pathogenic_idx]
    genmod_ids = hd5_file['legacy_genmod/id'][()]
    # Iterate trough genmod scores, to find pathogenic indexes in the genmod variant dataset
    genmod_causative_idx = []
    pbar = ProgressBar(max_value=len(causative_ids))
    pbar.start()
    for causative_id in causative_ids:
        for idx, genmod_id in enumerate(genmod_ids):
            if genmod_id == causative_id:
                genmod_causative_idx.append(idx)  # FIXME: Causes upsamling of genmod variants, since occur multiple times?
        pbar.increment(1)
    pbar.finish()
    genmod_causative_dataset = hd5_file['legacy_genmod'].create_dataset(
        name='causative',
        shape=(len(genmod_ids), ),
        dtype=np.float32,
    )
    genmod_causatives = np.zeros_like(genmod_ids, dtype=np.float32)
    genmod_causatives[genmod_causative_idx] = 1.0
    genmod_causative_dataset[()] = genmod_causatives[()]
    hd5_file.flush()
    hd5_file.close()


def compute_causative_rank(hd5_file_path: str,
                           output_file_path: str,
                           filter_variants_on_frequency_threshold: Union[float, None] = None):
    """
    Per case: find rank position of pathogenic variant.

    Additionally, filter MIVMIR variants on population frequency.
    TODO: Filter genmod variants on frequency as well?

    Aggregated metrics:
    - Rank of pathogenic variant (per case)
        - Genmod legacy
        - MIVMIR

    :param hd5_file_path: Path to HD5 containing input data
    :param output_file_path: Storage path for rank result file .CSV
    :param filter_variants_on_frequency_threshold: Population frequency threshold to exclude > filter_frq
    """
    hd5_file = Hd5File(hd5_file_path, 'r')
    data = {
        'id': hd5_file['gicamdata/id'][()],
        'case': hd5_file['gicamdata/case'][()],
        'score_mivmir': hd5_file['gicamdata/mivmir'][()],
        'score_gicam': hd5_file['gicamdata/gicam'][()],
        'causative': hd5_file['gicamdata/causative'][()]
    }
    if filter_variants_on_frequency_threshold:
        for frq_field_name in FRQ_FIELDS:
            data.update({frq_field_name: hd5_file[f"gicamdata/{frq_field_name}"][()]})
    df = pd.DataFrame(data=data)
    df.set_index('id', inplace=True)

    df_genmod = pd.DataFrame(
        data = {
            'id': hd5_file['legacy_genmod/id'][()],
            'score_legacy_genmod': hd5_file['legacy_genmod/genmod'][()],
            'causative': hd5_file['legacy_genmod/causative'][()],
        }
    )
    df_genmod.set_index('id', inplace=True)

    # Assemble DF with case as index, and rank score per mivmir and genmod legacy as columns
    ranked_cases: List[int] = []
    mivmir_ranks: List[int] = []
    genmod_ranks: List[int] = []
    gicam_ranks: List[int] = []
    causative_variant_ids: List[bytes] = []
    case_ids = df.case.unique()
    pbar = ProgressBar(max_value=len(case_ids))
    _LOGGER.info(f"Computing ranks, filter threshold={filter_variants_on_frequency_threshold}")
    pbar.start()
    for case in case_ids:
        case_variants = df[df.case == case].copy()
        # Additionally filter variants on population frequency
        if filter_variants_on_frequency_threshold:
            pop_frqs = case_variants[FRQ_FIELDS]
            pop_frqs = pop_frqs.fillna(0.0)  # Fill unseen variant frequencies with 0.0 (unseen frq)
            low_frq_variant_idx = pop_frqs[pop_frqs.max(axis=1) <= filter_variants_on_frequency_threshold].index.copy()
            del pop_frqs
            case_variants = case_variants.loc[low_frq_variant_idx]
        # Find all genmod variants that's present in the case_variants
        # FIXME: Multiple genmod variants per case, ranked multiple times because no case_id to separate them
        # len(case_variants_by_genmod) >> len(case_variants)
        case_variants_by_genmod = df_genmod.loc[case_variants.index].copy()
        # Deduce ranks of causative variant
        causative_variant_ids_mivmir_gicam = case_variants[case_variants.causative == 1].index
        # FIXME: There might be more causative  genmod variants than variants in MIVMIR set.
        # i.e. multiple marked pathogenic variants occur in the genmod set, if we lookup with mivmir variant IDs.
        causative_variant_ids_genmod = case_variants_by_genmod[case_variants_by_genmod.causative == 1].index
        # Assume only 1 causative variant per case for now
        common_causative_variant_ids = list(set(causative_variant_ids_mivmir_gicam.values).intersection(set(causative_variant_ids_genmod.values)))
        assert len(common_causative_variant_ids) <= 1, f"Expected only one (or none) causative variant: {common_causative_variant_ids}"
        if len(common_causative_variant_ids) == 0:
            mivmir_ranks.append(None)
            genmod_ranks.append(None)
            gicam_ranks.append(None)
            causative_variant_ids.append(None)
            ranked_cases.append(case)
            continue
        # Now causative variant is found across both datasets, compute the rank
        causative_variant_id = common_causative_variant_ids[0]
        rank_mivmir: int = case_variants.sort_values('score_mivmir', ascending=False, kind='stable').index.get_loc(causative_variant_id)
        rank_gicam: int = case_variants.sort_values('score_gicam', ascending=False, kind='stable').index.get_loc(causative_variant_id)
        rank_genmod: Union[int, List[bool]] = case_variants_by_genmod.sort_values('score_legacy_genmod', ascending=False, kind='stable').index.get_loc(causative_variant_id)
        if not isinstance(rank_genmod, (int, float)):  # ... it's a list of boolean indexes, [True, False, ... ]
            # In case genmod data contains multiple, causative variants (in this case), select the highest score
            # This is the first position in the index, starting a 0 and counting, which matches TRUE
            for rank_position, is_pathogenic in enumerate(rank_genmod):
                if is_pathogenic:
                    rank_genmod = rank_position
                    break
        assert isinstance(rank_mivmir, (int, float)), rank_mivmir
        mivmir_ranks.append(rank_mivmir)
        assert isinstance(rank_gicam, (int, float)), rank_gicam
        gicam_ranks.append(rank_gicam)
        assert isinstance(rank_genmod, (int, float)), rank_genmod
        genmod_ranks.append(rank_genmod)
        causative_variant_ids.append(causative_variant_id)
        ranked_cases.append(case)

        # TODO: Store to .hd5 dataset instead of .CSV
        rank_results = pd.DataFrame(
            data={
                'case_id': ranked_cases,
                'id': causative_variant_ids,
                'rank_mivmir': mivmir_ranks,
                'rank_gicam': gicam_ranks,
                'rank_legacy_genmod': genmod_ranks,
                # NOTE: variant_filter_frq is assumed to be identical, see downstream visualisation step
                'variant_filter_frq': np.full_like(ranked_cases,
                                                   fill_value=filter_variants_on_frequency_threshold,
                                                   dtype=np.float32)
            }
        )
        rank_results.to_csv(output_file_path)

        pbar.increment(1)
        del case_variants
        del case_variants_by_genmod
        gc.collect()
    pbar.finish()
    _LOGGER.info(f"Completed rank analysis, stored causative rank file in {output_file_path}")
    return output_file_path


def _plot_casewide_performance_metrics(hd5_file_path: str,
                                       storage_dir: str):

    # NOTE: The genmod and mivmir data points differ, theyre multiples in Genmod!

    hd5_file = Hd5File(hd5_file_path, 'r')
    data = {
        'score_mivmir': hd5_file['gicamdata/mivmir'][()],
        'score_gicam': hd5_file['gicamdata/gicam'][()],
        'causative': hd5_file['gicamdata/causative'][()]
    }
    df = pd.DataFrame(data=data)

    df_genmod = pd.DataFrame(
        data={
            'score_legacy_genmod': hd5_file['legacy_genmod/genmod'][()],
            'causative': hd5_file['legacy_genmod/causative'][()],
        }
    )
    hd5_file.close()

    # Boxplot, violinplot of inference values for benign, causative variants
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 2, 1)
    box_data = [
        df[df.causative != LABEL_PATHOGENIC_VARIANT].score_mivmir.values,
        df[df.causative != LABEL_PATHOGENIC_VARIANT].score_gicam.values,
        df_genmod[df_genmod.causative != LABEL_PATHOGENIC_VARIANT].score_legacy_genmod.values,
        df[df.causative == LABEL_PATHOGENIC_VARIANT].score_mivmir.values,
        df[df.causative == LABEL_PATHOGENIC_VARIANT].score_gicam.values,
        df_genmod[df_genmod.causative == LABEL_PATHOGENIC_VARIANT].score_legacy_genmod.values
    ]
    ax.boxplot(box_data)
    ax.set_xticks([1, 2, 3, 4, 5, 6], labels=['MIVMIR [benign]',
                                        'GICAM [benign]',
                                        'GENMOD [benign]',
                                        'MIVMIR [causative]',
                                        'GICAM [causative]',
                                        'GENMOD [causative]'])
    ax.yaxis.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.violinplot(box_data,
                  showmeans=False,
                  showmedians=True)
    ax.yaxis.grid(True)
    ax.set_xticks([1, 2, 3, 4, 5, 6], labels=['MIVMIR [benign]',
                                        'GICAM [benign]',
                                        'GENMOD [benign]',
                                        'MIVMIR [causative]',
                                        'GICAM [causative]',
                                        'GENMOD [causative]'])
    fig.suptitle('Inference values')
    fig.savefig(os.path.join(storage_dir, 'inference-values.png'))

    # Mivmir performance
    _LOGGER.info("Generating MIVMIR classifier plots")
    statfns.plot_roc_auc(predictions=df.score_mivmir.values,
                         truths=df.causative.values,
                         pos_label=LABEL_PATHOGENIC_VARIANT,
                         output_path=os.path.join(storage_dir, 'mivmir-roc-auc.png'))
    statfns.confusion_matrix(predictions=df.score_mivmir.values,
                             truths=df.causative.values,
                             discretisation_threshold=0.5,
                             output_path=os.path.join(storage_dir, 'mivmir-confusion-matrix.png'))
    statfns.plot_performance_vs_threshold(predictions=df.score_mivmir.values,
                                          labels=df.causative.values,
                                          n_steps=75,
                                          output_path=os.path.join(storage_dir, 'mivmir-performance.png'))

    # GICAM performance
    _LOGGER.info("Generating GICAM classifier plots")
    statfns.plot_roc_auc(predictions=df.score_gicam.values,
                         truths=df.causative.values,
                         pos_label=LABEL_PATHOGENIC_VARIANT,
                         output_path=os.path.join(storage_dir, 'gicam-roc-auc.png'))
    statfns.confusion_matrix(predictions=df.score_gicam.values,
                             truths=df.causative.values,
                             discretisation_threshold=0.5,
                             output_path=os.path.join(storage_dir, 'gicam-confusion-matrix.png'))
    statfns.plot_performance_vs_threshold(predictions=df.score_gicam.values,
                                          labels=df.causative.values,
                                          n_steps=75,
                                          output_path=os.path.join(storage_dir, 'gicam-performance.png'))

    # Genmod performance
    _LOGGER.info("Generating Genmod classifier plots")
    statfns.plot_roc_auc(predictions=df_genmod.score_legacy_genmod.values,
                         truths=df_genmod.causative.values,
                         pos_label=LABEL_PATHOGENIC_VARIANT,
                         output_path=os.path.join(storage_dir, 'genmod-roc-auc.png'))
    statfns.confusion_matrix(predictions=df_genmod.score_legacy_genmod.values,
                             truths=df_genmod.causative.values,
                             discretisation_threshold=0.5,
                             output_path=os.path.join(storage_dir, 'genmod-confusion-matrix.png'))
    statfns.plot_performance_vs_threshold(predictions=df_genmod.score_legacy_genmod.values,
                                          labels=df_genmod.causative.values,
                                          output_path=os.path.join(storage_dir, 'genmod-performance.png'))


def visualize_performance(rank_results_file_path: str,
                          case_id_to_name_map_path: str,
                          tmp_storage_dir: str,
                          adjust_for_borked_genmod_scores: bool = False):
    df = pd.read_csv(rank_results_file_path)
    df_case_ids = pd.read_csv(case_id_to_name_map_path, sep='\t')
    df_case_ids.index = df_case_ids.Line_Number
    df_case_ids.sort_index(inplace=True)

    # Translate case ID:int into string (actual case name)
    case_names = []
    for case_id in df.case_id.values:
        case_names.append(df_case_ids.loc[case_id].Sample_ID)
    df['case_name'] = case_names
    del df_case_ids

    if adjust_for_borked_genmod_scores:
        rank_legacy_genmod = []
        for row in df.rank_legacy_genmod:
            try:
                rank_legacy_genmod.append(int(row))
            except ValueError:
                rank_legacy_genmod.append(None)
        df.rank_legacy_genmod = rank_legacy_genmod

    # Select data to plot
    plot_data = df[['rank_mivmir', 'rank_gicam', 'rank_legacy_genmod', 'case_name']].copy()
    plot_data.set_index('case_name', inplace=True)
    plot_data.sort_index(inplace=True)
    plot_data = plot_data.dropna()  # Plots cannot handle NaNs
    if len(plot_data) < len(df):
        _LOGGER.warning(f"Dropped {len(df) - len(plot_data)} samples from plotted data due to NaNs")

    # Get population filter frequency in case variants were filtered
    variant_frq_filter = (df.variant_filter_frq.unique())[0]  # Assumed all identical

    points_mivmir = len(plot_data[plot_data.rank_mivmir < plot_data.rank_legacy_genmod])
    points_gicam = len(plot_data[plot_data.rank_gicam < plot_data.rank_legacy_genmod])
    points_genmod = len(plot_data[plot_data.rank_mivmir > plot_data.rank_legacy_genmod])
    points_tie = len(plot_data[plot_data.rank_mivmir == plot_data.rank_legacy_genmod])
    points_tie_gicam = len(plot_data[plot_data.rank_gicam == plot_data.rank_legacy_genmod])

    # Plot rank case by case
    def _plot_cases(case_names: List[str], chunk_idx):
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot()
        ax.scatter(x=plot_data.loc[case_names].index,
                   y=plot_data.loc[case_names].rank_legacy_genmod,
                   marker='o',
                   alpha=0.75)
        ax.scatter(x=plot_data.loc[case_names].index,
                   y=plot_data.loc[case_names].rank_mivmir,
                   marker='D',
                   alpha=0.75)
        ax.scatter(x=plot_data.loc[case_names].index,
                   y=plot_data.loc[case_names].rank_gicam,
                   marker='s',
                   alpha=0.5)
        ax.grid(True, which='both')
        ax.minorticks_on()
        plt.xticks(rotation=45)
        ax.legend(['Rank Legacy Genmod', 'Rank MIVMIR', 'Rank GICAM'])
        fig.tight_layout()
        fig.savefig(os.path.join(tmp_storage_dir, f"case_ranks-{chunk_idx}.png"))

    def _to_chunks(case_ids: list, chunk_size: int):
        for i in range(0, len(case_ids), chunk_size):
            yield case_ids[i: i + chunk_size]

    caseid_chunks = _to_chunks(plot_data.index.to_list(), 25)
    for chunk_idx, caseid_chunk in enumerate(caseid_chunks):
        _plot_cases(caseid_chunk, chunk_idx)

    # Violin, box-whisker and swarm plots side by side
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 3, 1)
    ax.violinplot(plot_data,
                  showmeans=False,
                  showmedians=True)
    ax.yaxis.grid(True)
    ax.set_ylabel('Pathogenic Variant Rank Position')
    ax.set_xticks(range(1, len(plot_data.columns.values) + 1), labels=plot_data.columns.values)
    ax = fig.add_subplot(1, 3, 2)
    ax.boxplot(plot_data)
    ax.set_xticks(range(1, len(plot_data.columns.values) + 1), labels=plot_data.columns.values)
    ax.yaxis.grid(True)
    ax = fig.add_subplot(1, 3, 3)
    sb.swarmplot(plot_data, size=3, ax=ax)
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.suptitle(f"filt_frq:{variant_frq_filter}\nPoints Mivmir:{points_mivmir}\nPoints GICAM: {points_gicam}\nPoints Genmod:{points_genmod}\nTie MIVMIR:{points_tie}, Tie GICAM {points_tie_gicam}\nn={len(plot_data)}/{len(df)}")
    fig.savefig(os.path.join(tmp_storage_dir, 'rank-stats.png'))

    # Scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=plot_data.rank_mivmir, y=plot_data.rank_legacy_genmod)
    ax.set_xlabel('Rank Mivmir')
    ax.set_ylabel('Rank Genmod')
    fig.tight_layout()
    fig.suptitle('Scatter plot of ranks')
    ax.grid(True)
    fig.savefig(os.path.join(tmp_storage_dir, 'rank-scatter-mivmir-genmod.png'))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x=plot_data.rank_gicam, y=plot_data.rank_legacy_genmod)
    ax.set_xlabel('Rank GICAM')
    ax.set_ylabel('Rank Genmod')
    fig.tight_layout()
    fig.suptitle('Scatter plot of ranks')
    ax.grid(True)
    fig.savefig(os.path.join(tmp_storage_dir, 'rank-scatter-gicam-genmod.png'))

def _compute_and_visualize_rank(hd5_file_path,
                                tmp_storage_dir,
                                filter_variants_on_frequency_threshold,
                                case_id_to_name_map):
    suffix = '-nofilt' if filter_variants_on_frequency_threshold is None else f"-{filter_variants_on_frequency_threshold}"
    ranked_output_file_path = os.path.join(tmp_storage_dir, f"rank{suffix}.csv")
    file_containing_case_causative_ranks = compute_causative_rank(hd5_file_path=hd5_file_path,
                                                                  output_file_path=ranked_output_file_path,
                                                                  filter_variants_on_frequency_threshold=filter_variants_on_frequency_threshold)
    # Writes images based on frq filt
    sub_dir_path = os.path.join(tmp_storage_dir, os.path.basename(file_containing_case_causative_ranks).replace('.csv', ''))
    os.mkdir(sub_dir_path)
    visualize_performance(rank_results_file_path=file_containing_case_causative_ranks,
                          case_id_to_name_map_path=case_id_to_name_map,
                          tmp_storage_dir=sub_dir_path)


def build_analyze_mivmir_nextflow_dataset(mivmir_scores_csv: str,
                                          default_genmod_csv: str,
                                          case_id_to_name_map: str,
                                          output_file_path: str,):
    # Create tmpdir where to store images
    tmpdir_object = tempfile.TemporaryDirectory(dir='/tmp')
    tmp_storage_dir = tmpdir_object.name

    hd5_file_path = convert_csv_to_hd5(mivmir_scores_csv=mivmir_scores_csv,
                                       default_genmod_csv=default_genmod_csv,
                                       output_file_path=output_file_path)
    find_causative_genmod_variants(hd5_file_path=hd5_file_path)

    _plot_casewide_performance_metrics(hd5_file_path=output_file_path,
                                       storage_dir=tmp_storage_dir)

    rare_frq = 1.0/2000.0
    kwargs = []
    for filter_frq in [None, rare_frq, 10*rare_frq, 100*rare_frq]:
        kwargs.append({
            'hd5_file_path': hd5_file_path,
            'tmp_storage_dir': tmp_storage_dir,
            'filter_variants_on_frequency_threshold': filter_frq,
            'case_id_to_name_map': case_id_to_name_map
        })
    pool = ProcessPool(function=_compute_and_visualize_rank,
                       kwargs=kwargs)
    completed_tasks = pool.run()
    for task in completed_tasks:
        assert task.process.exitcode == 0, task

    # Create archive of all plots
    archive_path = output_file_path.replace('.hd5', '-plots')
    assert archive_path != output_file_path
    shutil.make_archive(base_name=archive_path,
                        format='tar',
                        root_dir=tmp_storage_dir,
                        logger=_LOGGER)

    _LOGGER.info("Completed")
