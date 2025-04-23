import pytest as pt
import tensorflow as tf

from rdds.lib.tf import DnaSequenceTrimmer


@pt.fixture
def sequences():
    return tf.ragged.constant([[b'foo  ', b'bar', b'cctaggcnt', b'barc', b'c'],
                               [b'pathogenic', b'CAT', b'0.32']], dtype=tf.string)


def test_dna_sequence_trimmer(sequences):
    """
    Test to check trimming of DNA sequences from RaggedTensor input.
    """
    # GIVEN a trimmer instance
    trimmer = DnaSequenceTrimmer()
    # WHEN trimming the ragged tensor sequences
    r = trimmer(sequences)
    assert isinstance(r, tf.RaggedTensor)
    r = r.to_list()
    b0 = set(r[0])
    b1 = set(r[1])
    # THEN expect the DNA sequences to be removed, remaining untouched.
    assert b0.difference({b'foo  ', b'bar', b'barc'}) == set()
    assert b1.difference({b'pathogenic', b'0.32'}) == set()
