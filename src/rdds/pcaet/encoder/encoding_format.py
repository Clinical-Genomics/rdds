from dataclasses import dataclass, field


@dataclass(repr=True)
class EncodingFormat:
    genes: str = field(default='')
    # phenotyp som HPO termer
    patient_phenotype: str = field(default='')
    relevance_for_clinical_evaluation_of_pathogenicity: str = field(default='')
    # MIVMIR score
    # VEP consequence