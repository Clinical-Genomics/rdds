def test_agent():
    from rdds.pcaet.encoder.attribute_relevance_agent import VariantAttributeRelevanceAgent

    agent = VariantAttributeRelevanceAgent()
    result = agent.infer_relevance_of_variant_field_to_format_keyword(variant_vcf_info_attribute='CADD',
                                                                      encoding_format_keyword='pathogenicity')
    result
