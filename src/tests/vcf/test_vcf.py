from rdds.lib.vcf import VCFReader


def test_load_file(minimally_annotated_ranked_mip_vcf):
    """
    Test for loading VCF file.
    """
    # GIVEN a VCF file
    # WHEN reading it
    vcf_reader: VCFReader = VCFReader(minimally_annotated_ranked_mip_vcf)
    assert vcf_reader.number_of_variants > 0
    for i in vcf_reader:
        # THEN expect the first row to be readable and contain expected data
        assert i.INFO['RankScore'] == '1:8', i.INFO['RankScore']
        break
    # THEN expect vcf reader to find available data fields
    assert vcf_reader.data_fields == ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'MQ', 'Annotation', 'Exonic', '1000GAF',
                                      'CADD', 'GeneticModels', 'ModelScore', 'Compounds', 'RankScore', 'GT', 'AD', 'GQ']


def test_reader_static_vcf_fields(minimally_annotated_ranked_mip_vcf):
    """
    Test for checking the contents of str-Enum type StaticVcfFields
    """
    # GIVEN a vcf reader
    vcf_reader = VCFReader(minimally_annotated_ranked_mip_vcf)
    # WHEN available
    # THEN expect the static data fields to contain a List[str] with proper content
    assert isinstance(vcf_reader.static_data_fields, list)
    assert vcf_reader.static_data_fields == ['CHROM', 'POS', 'ID', 'REF', 'ALT']


def test_amount_read_from_vcf(fully_annotated_unranked_mip_vcf):
    """
    Test for reading and parsing variants in VCF file.
    """
    # GIVEN a VCF file
    # WHEN reading it
    expected_nr_variants, vcf_file = fully_annotated_unranked_mip_vcf
    vcf_reader: VCFReader = VCFReader(vcf_file)
    assert vcf_reader.number_of_variants == expected_nr_variants
    for n_variants, unparsed_variant in enumerate(vcf_reader, start=1):
        continue
        # THEN expect all variants to be read
    assert n_variants == expected_nr_variants