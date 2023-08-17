from os.path import dirname
EXAMPLE_FILE: str = dirname(__file__)+'/test.vcf.gz'

from rdds.lib.vcf import VCFReader


def test_load_file():
    """
    Test for loading VCF file.
    """
    # GIVEN a VCF file
    # WHEN reading it
    vcf_reader: VCFReader = VCFReader(EXAMPLE_FILE)
    assert vcf_reader.number_of_variants > 0
    for i in vcf_reader:
        # THEN expect the first row to be readable and contain expected data
        assert i.INFO['RankScore'] == '1:8', i.INFO['RankScore']
        break
    # THEN expect vcf reader to find available data fields
    assert vcf_reader.data_fields == ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'MQ', 'Annotation', 'Exonic', '1000GAF',
                                      'CADD', 'GeneticModels', 'ModelScore', 'Compounds', 'RankScore', 'GT', 'AD', 'GQ']


def test_reader_static_vcf_fields():
    """
    Test for checking the contents of str-Enum type StaticVcfFields
    """
    # GIVEN a vcf reader
    vcf_reader = VCFReader(EXAMPLE_FILE)
    # WHEN available
    # THEN expect the static data fields to contain a List[str] with proper content
    assert isinstance(vcf_reader.static_data_fields, list)
    assert vcf_reader.static_data_fields == ['CHROM', 'POS', 'ID', 'REF', 'ALT']
