from os.path import dirname
EXAMPLE_FILE: str = dirname(__file__)+'/test_mip.vcf.gz'

from rdds.lib.vcf import VCFReader, ParsableVariant

_VCF_LENGTH = 500
_VCF_HEADER_LENGTH = 110
_VCF_NR_VARIANTS = _VCF_LENGTH - _VCF_HEADER_LENGTH


def test_parsable_variant():
    """
    Test for reading and parsing variants in VCF file.
    """
    # GIVEN a VCF file
    # WHEN reading it
    vcf_reader: VCFReader = VCFReader(EXAMPLE_FILE)
    assert vcf_reader.number_of_variants == _VCF_NR_VARIANTS
    csq_description: str = vcf_reader.csq_description
    for n_variants, unparsed_variant in enumerate(vcf_reader):
        _ = ParsableVariant(variant=unparsed_variant,
                            vep_csq_description=csq_description)
        # THEN expect the variants to be parsed without error
    assert n_variants + 1 == _VCF_NR_VARIANTS  # indexing starts at 0
