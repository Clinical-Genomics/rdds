import pytest as pt

from ...fixtures import BRCA_ARTICLE
from ...fixtures import AMUSING_MARMORSET_VARIANTS

def test_encode_document():
    from rdds.pcaet.encoder import DocumentEncoder, Document

    document_encoder = DocumentEncoder()
    document = Document(BRCA_ARTICLE)
    encoding_result = document_encoder.encode(document)

    assert 'BRCA1' in encoding_result.input.genes
    assert 'BRCA2' in encoding_result.input.genes
    assert 'breast cancer' in encoding_result.input.patient_phenotype
    assert 'early-onset' in encoding_result.input.patient_phenotype.lower()

def test_encode_patient_case():
    from rdds.lib.vcf import VCFReader
    from rdds.pcaet.encoder import VariantEncoder, ParsableVariant

    vcf_reader = VCFReader(AMUSING_MARMORSET_VARIANTS, 'r')
    vcf_header: str = vcf_reader.raw_header

    variant_encoder = VariantEncoder(vcf_header=vcf_header)

    variants = list(vcf_reader)
    variants = variants[0:1]
    parsed_variants = [ParsableVariant(variant=variant, vep_csq_description=vcf_reader.csq_description) \
                       for variant in variants]
    for parsed_variant in parsed_variants:
        encoding_result = variant_encoder.encode(variant=parsed_variant)
        print(encoding_result)
