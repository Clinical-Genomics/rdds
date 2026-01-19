import pandas as pd

_HPO_PHEN_TO_GENE_TSV = '/rdds/tmp/dataset-hpo/phenotype_to_genes.txt'
_HPO_GENES_TO_DISEASE = '/rdds/tmp/dataset-hpo/genes_to_disease.txt'
_HPO_FREQUENCY_TO_DISEASE = '/rdds/tmp/dataset-hpo/phenotype.hpoa'

class Phen2GenDatasetCompiler:

    def __init__(self,
                 hpo_phen_to_gene_tsv: str = _HPO_PHEN_TO_GENE_TSV,
                 hpo_genes_to_disease: str = _HPO_GENES_TO_DISEASE,
                 hpo_frequency_to_disease: str = _HPO_FREQUENCY_TO_DISEASE):
        self._hpo_phen_to_gene_tsv = hpo_phen_to_gene_tsv
        self._hpo_genes_to_disease_tsv = hpo_genes_to_disease
        self._hpo_frequency_to_disease = hpo_frequency_to_disease

    def compile(self):
        on_bad_lines = "error"
        # Prepare HPO Phenotype to Genotype TSV
        df_phenotype_to_genes = pd.read_csv(self._hpo_phen_to_gene_tsv,
                                            low_memory=False,
                                            on_bad_lines=on_bad_lines,
                                            delimiter='\t')

        # Prepare HPO phenotype ID to NCBI gene ID
        df_genes_to_disease = pd.read_csv(self._hpo_genes_to_disease_tsv,
                                          low_memory=False,
                                          on_bad_lines=on_bad_lines,
                                          delimiter='\t')

        # Prepare OMIM HPO Frequency to Disease mappings
        df_frequency_to_disease = pd.read_csv(self._hpo_frequency_to_disease,
                                              header=4,
                                              on_bad_lines=on_bad_lines,
                                              delimiter='\t')
        ## Drop all not of datatype 'aspect:phenotypic abnormality', https://hpo.jax.org/browse/term/HP:0000118
        df_frequency_to_disease = df_frequency_to_disease[df_frequency_to_disease.aspect == 'P']
        # TODO: Decode frequency term according to https://obophenotype.github.io/human-phenotype-ontology/annotations/frequency/
        # TODO: Make use of 'qualifier' and NOT annotation for negative associations
        return