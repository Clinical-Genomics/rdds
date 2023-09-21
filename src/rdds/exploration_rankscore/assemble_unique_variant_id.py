import pandas as pd
import numpy as np
from typing import List


def assemble_unique_variant_id(chromosome_pos: int,
                               variant_id: bytes,
                               alt_allele: bytes,
                               ref_allele: bytes) -> bytes:
    """
    Compose a variant bytestring, that's a composition of VCF POS, ID, ALT, REF.
    This is necessary since the VCF.ID can be shared among a set of variants in case there are more
    variants at the genomic position, but with different ALT mutations.

    :param chromosome_pos: VCF Chromosomal position
    :param variant_id: VCF ID field for variant
    :param alt_allele: VCF ALT field for variant
    :param ref_allele: VCF REF field for variant
    :return: bytestring, unique composition of ID, ALT, REF
    """
    return b'%d' % chromosome_pos + b'-' + variant_id + b'-alt-' + alt_allele + b'-ref-' + ref_allele


def dataframe_assemble_unique_variant_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex dataframe containing variants with a new unique index, based
    on the VCF.ID, ALT, REF fields.

    :param df: Dataframe containing at least ID (current index), ALT, REF fields (lowercase)
    :return: Reindexed DF
    :raises ValueError: In case attempting to create duplicate UID
    """
    df = df.copy()  # Don't modify original DF
    unique_ids: List[bytes] = []
    ids: np.ndarray = df.index.values
    positions: np.ndarray = df.pos.values
    alts: np.ndarray = df.alt.values
    refs: np.ndarray = df.ref.values
    for pos, id, alt, ref in zip(positions, ids, alts, refs):
        unique_id = assemble_unique_variant_id(chromosome_pos=pos, variant_id=id, alt_allele=alt, ref_allele=ref)
        unique_ids.append(unique_id)
    unique_ids_df = pd.DataFrame(data={'UID': unique_ids}, index=ids)
    df_reindexed = pd.concat((df, unique_ids_df), axis=1)
    df_reindexed['id'] = df_reindexed.index
    df_reindexed = df_reindexed.set_index('UID')
    if not df_reindexed.index.is_unique:
        raise ValueError(f'Duplicate UIDs in dataframe: {df_reindexed[df_reindexed.index.duplicated()]}')
    return df_reindexed
