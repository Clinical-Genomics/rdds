import pytest
from rdds.lib.vcf import VCFReader, ParsableVariant

_VCF_LENGTH = 500
_VCF_HEADER_LENGTH = 110
_VCF_NR_VARIANTS = _VCF_LENGTH - _VCF_HEADER_LENGTH


def test_parsable_variant_scored(minimally_annotated_ranked_mip_vcf):
    """
    Test for parsing RankScore field as annotated by MIP pipeline.
    """
    # GIVEN a VCF containing rank scores
    # WHEN reading and parsing the variants
    vcf_reader: VCFReader = VCFReader(minimally_annotated_ranked_mip_vcf)
    for unparsed_variant in vcf_reader:
        variant = ParsableVariant(variant=unparsed_variant,
                                  vep_csq_description=vcf_reader.csq_description)
        continue
        # THEN expect them to contain data
        assert isinstance(variant.RankScore_value, float) and 11 >= variant.RankScore_value >= -20
        # RankScore_family_id is fixed in this VCF file
        assert isinstance(variant.RankScore_family_id, float) and variant.RankScore_family_id == 1


def test_parsable_variant(fully_annotated_unranked_mip_vcf):
    """
    Test for reading and parsing variants in VCF file as annotated by MIP pipeline.
    """
    # GIVEN a VCF file
    # WHEN reading it
    _, fully_annotated_unranked_mip_vcf = fully_annotated_unranked_mip_vcf
    vcf_reader: VCFReader = VCFReader(fully_annotated_unranked_mip_vcf)

    parsed_csq_fields = set()  # Placeholder to monitor parsed VEP INFO/CSQ fields

    for unparsed_variant in vcf_reader:
        variant = ParsableVariant(variant=unparsed_variant,
                                  vep_csq_description=vcf_reader.csq_description)
        # THEN expect the variant meta data to be parsed
        assert type(variant.CHROM) == str and variant.CHROM != ''
        assert type(variant.POS) == int and variant.POS > 0
        assert type(variant.ID) == str and variant.ID != ''
        assert type(variant.REF) == str and variant.REF != ''
        assert type(variant.ALT) == str and variant.ALT != ''
        with pytest.raises(AttributeError):
            # THEN Expect INFO not to be parsed as a str field (should be unpacked)
            variant.INFO
        # Optional fields (not always present)
        if 'CADD' in variant.parsed_fields:
            assert type(variant.CADD) == float and variant.CADD >= 0
        for csq_sub_field in vcf_reader.csq_sub_fields:
            # THEN expect INFO/CSQ_[FIELD] to be parsed, if present
            if csq_sub_field in variant.parsed_fields:
                parsed_csq_fields.add(csq_sub_field)
                assert type(variant.__getattribute__(csq_sub_field)) in [str, int, float]
        if 'Frq' in variant.parsed_fields:
            assert type(variant.Frq) == float and variant.Frq > 0
        if 'GNOMADAF' in variant.parsed_fields:
            assert type(variant.GNOMADAF) == float and variant.GNOMADAF > 0
        if 'GNOMADAF_popmax' in variant.parsed_fields:
            assert type(variant.GNOMADAF_popmax) == float and variant.GNOMADAF_popmax > 0
        if 'GNOMADAF_MTAF_het' in variant.parsed_fields:
            assert type(variant.GNOMADAF_MTAF_het) == float and variant.GNOMADAF_MTAF_het > 0
        if 'GNOMADAF_MTAF_hom' in variant.parsed_fields:
            assert type(variant.GNOMADAF_MTAF_hom) == float and variant.GNOMADAF_MTAF_hom > 0
        if 'Hom' in variant.parsed_fields:
            assert type(variant.Hom) == float and variant.Hom > 0
        if 'MTAF' in variant.parsed_fields:
            assert type(variant.MTAF) == float and variant.MTAF > 0
        if 'Obs' in variant.parsed_fields:
            assert type(variant.Obs) == float and variant.Obs > 0
        if 'SPIDEX' in variant.parsed_fields:
            assert type(variant.SPIDEX) == float and variant.SPIDEX > 0
        if 'SWEGENAAC_Hemi' in variant.parsed_fields:
            assert type(variant.SWEGENAAC_Hemi) == float and variant.SWEGENAAC_Hemi >= 0
        if 'SWEGENAAC_Het' in variant.parsed_fields:
            assert type(variant.SWEGENAAC_Het) == float and variant.SWEGENAAC_Het >= 0
        if 'SWEGENAF' in variant.parsed_fields:
            assert type(variant.SWEGENAF) == float and variant.SWEGENAF > 0
        if 'genomic_superdups_frac_match' in variant.parsed_fields:
            assert type(variant.genomic_superdups_frac_match) == float and variant.genomic_superdups_frac_match > 0
        if 'mitotip_score' in variant.parsed_fields:
            assert type(variant.mitotip_score) == float and variant.mitotip_score >= 0
        if 'mitotip_trna_prediction' in variant.parsed_fields:
            assert type(variant.mitotip_trna_prediction) == str and variant.mitotip_trna_prediction != ''
        if 'most_severe_consequence' in variant.parsed_fields:
            assert type(variant.most_severe_consequence) == str and variant.most_severe_consequence != ''
        if 'Annotation' in variant.parsed_fields:
            assert type(variant.Annotation) == str and variant.Annotation != ''
        if 'GeneticModels_family_id' in variant.parsed_fields:
            assert type(variant.GeneticModels_family_id) == str and variant.GeneticModels_family_id != ''
        if 'GeneticModels_model' in variant.parsed_fields:
            assert type(variant.GeneticModels_model) == str and variant.GeneticModels_model != ''
        if 'ModelScore_value' in variant.parsed_fields:
            assert type(variant.ModelScore_value) == float and variant.ModelScore_value > 0
        if 'ModelScore_family_id' in variant.parsed_fields:
            assert type(variant.ModelScore_family_id) == str and variant.ModelScore_family_id != ''
        if 'Compounds' in variant.parsed_fields:
            assert type(variant.Compounds) == str and variant.Compounds != ''
        # Annotated labels test
        if 'CLINVAR_GROUND_TRUTH' in variant.parsed_fields:
            assert type(variant.CLINVAR_GROUND_TRUTH) == str and variant.CLINVAR_GROUND_TRUTH != ''
        if 'GIAB_GROUND_TRUTH' in variant.parsed_fields:
            assert type(variant.GIAB_GROUND_TRUTH) == str and variant.GIAB_GROUND_TRUTH != ''
        if 'MUTACC_GROUND_TRUTH' in variant.parsed_fields:
            assert type(variant.MUTACC_GROUND_TRUTH) == str and variant.MUTACC_GROUND_TRUTH != ''
    for csq_field in vcf_reader.csq_sub_fields:
        # THEN expect all VCF CSQ sub fields to have been parsed in at least one variant
        assert csq_field in parsed_csq_fields
