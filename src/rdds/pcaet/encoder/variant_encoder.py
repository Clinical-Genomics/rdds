from .encoder import Encoder, OllamaClient, EncodingFormat, EncodingResult
from rdds.lib.vcf import Variant, ParsableVariant
from rdds.pcaet.encoder.attribute_relevance_agent import VariantAttributeRelevanceAgent

# TODO: Add metadata on VCF fields to LLM as context
# VCF format spec: https://samtools.github.io/hts-specs/
# VEP plugin spec: https://grch37.ensembl.org/info/docs/tools/vep/script/vep_plugins.html

class VariantEncoder(Encoder):

    def __init__(self,
                 vcf_header: str,
                 *args,
                 **kwargs):
        """
        :param vcf_header: VCF header, this information is used for judging variant attribute
            relevance to the encoding step.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # TODO: Variant header iter, vcf_reader.header_iter
        self._variant_encoding_llm = OllamaClient()
        self._vcf_header = vcf_header

    def _variant_attribute_search(self, keyword: str, variant: ParsableVariant) -> str:
        agent = VariantAttributeRelevanceAgent()
        for variant_attribute in variant:
            is_relevant = agent.lookup(vcf_info_attribute=keyword)

    def encode(self, variant: ParsableVariant):
        """
        Encode Variant to embedding representation
        :param variant: A Variant instance
        :returns: Embeddings
        """
        # For the variant, find variant attributes relevant to encoding format attributes
        encoding_format = EncodingFormat()
        for encoding_format_key in vars(encoding_format):
            for variant_attribute in variant:
                agent = VariantAttributeRelevanceAgent()
                is_relevant = \
                    agent.infer_relevance_of_variant_field_to_format_keyword(variant_vcf_info_attribute=variant_attribute,
                                                                             encoding_format_keyword=encoding_format_key)
            encoding_format.__setattr__(encoding_format_key, variant[variant_attribute])

        # Encode
        return super().encode(encoding_format)