from os.path import dirname
EXAMPLE_FILE: str = dirname(__file__)+'/test.vcf'

from rdds.lib.vcf import VCFReader

def test_load_file():
    # GIVEN a VCF file
    # WHEN reading it
    vcf_reader: VCFReader = VCFReader(EXAMPLE_FILE)
    for i in vcf_reader:
        # THEN expect the first row to be readable and contain expected data
        assert i.INFO['RankScore'] == '1:8', i.INFO['RankScore']
        break
