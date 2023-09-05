from typing import Set, Dict

from .class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT

CLINVAR_CLNSIG_BENIGN_LABELS: Set[str] = {
    'benign',
    'likely_benign',
    'affects',  # For variants that cause a non-disease phenotype, such as lactose intolerance.
    'drug_response',  # A general term for a variant that affects a drug response, not a disease.
    'protective',  # For variants that decrease the risk of a disorder, including infections.

}
CLINVAR_CLNSIG_PATHOGENIC_LABELS: Set[str] = {
    'pathogenic',
    'likely_pathogenic',
    # For variants identified in a GWAS study and further interpreted for their clinical significance.
    'association',
    'risk_factor',
    'likely_risk_allele',
}

CLINVAR_CLNSIG_DROP_LABELS: Set[str] = {  # Data matching this criteria should be dropped
    '.',
    'uncertain_significance',
    'not_provided',
    'other',
    'confers_sensitivity',  # TODO: Clinvar description of this field, possibly add to benign/pathogenic list
    'association_not_found',  # TODO: Clinvar description of this field, possibly add to benign/pathogenic list
    'uncertain_risk_allele',  # TODO: Clinvar description of this field, possibly add to benign/pathogenic list
    'conflicting_interpretations_of_pathogenicity',
    'low_penetrance',
}

CLINVAR_LABEL_MAPPING: Dict[str, float] = {}
[CLINVAR_LABEL_MAPPING.update({label: LABEL_BENIGN_VARIANT}) for label in CLINVAR_CLNSIG_BENIGN_LABELS]
[CLINVAR_LABEL_MAPPING.update({label: LABEL_PATHOGENIC_VARIANT}) for label in CLINVAR_CLNSIG_PATHOGENIC_LABELS]
