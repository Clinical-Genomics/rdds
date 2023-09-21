import pandas as pd
import pytest as pt

from rdds.exploration_rankscore.assemble_unique_variant_id import assemble_unique_variant_id, dataframe_assemble_unique_variant_id


def test_assemble_unique_variant_id():
    """
    Test for creating a variant UID based on ID, ALT, REF VCF fields
    """
    # GIVEN a variant with POS, ID, ALT, and REF metadata
    # WHEN assembling a new UID
    new_id = assemble_unique_variant_id(chromosome_pos=1000, variant_id=b'10_1032245_CA_C', alt_allele=b'C', ref_allele=b'CA')
    # THEN expect it to be a correctly assembled binary string
    assert new_id == b'1000-10_1032245_CA_C-alt-C-ref-CA'


def test_assemble_unique_variant_ids_in_dataframe():
    """
    Test for updating a dataframe with unique IDs (UIDs)
    """
    # GIVEN a dataframe with some dummy variants
    df = pd.DataFrame(data={'pos': [0, 1], 'alt': [b'C', b'A'], 'ref': [b'A', b'C']}, index=[b'id0', b'id0'])
    # WHEN assembling, reindexing the DF based on new UID
    df = dataframe_assemble_unique_variant_id(df=df)
    # THEN return data frame with 2 UIDs
    assert len(df) == 2
    assert df.index[0] == b'0-id0-alt-C-ref-A'
    assert df.index[1] == b'1-id0-alt-A-ref-C'


def test_duplicate_uid_error():
    """
    Test for making sure duplicate UIDs are caught.
    """
    # GIVEN a dataframe with two variants with identical ID, ALT, REF
    df = pd.DataFrame(data={'pos': [0, 0], 'alt': [b'C', b'C'], 'ref': [b'A', b'A']}, index=[b'id0', b'id0'])
    with pt.raises(ValueError):
        # WHEN reindexing
        dataframe_assemble_unique_variant_id(df=df)
        # THEN expect duplicates to be caught (error raised)
