# This file contains the data schema definition
# https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/schema.md

_SCHEMA = """
node_sets {
    key: "hpo"
    value {
        description: "A HPO term"

        features {
            key: "hpo_id"
            value: {
                description: "ID of the HPO term, like HP:0030992 without the HPO prefix"
                dtype: DT_INT64
            }
        }

        features {
            key: "hpo_name"
            value: {
                description: "HPO human readable name as bag of words, like Abnormal pancratic duct morphology"
                dtype: DT_STRING
                shape { dim { size: -1 } }
            }
        }
    }
}

node_sets {
    key: "gene"
    value {
        description: "A gene"

        features {
            key: "gene_id"
            value: {
                description: "NCBI gene ID"
                dtype: DT_INT64
            }
        }

        features {
            key: "gene_symbol"
            value: {
                description: "NCBI gene symbol"
                dtype: DT_STRING
                shape { dim { size: -1 } }
            }
        }
    }
}

node_sets {
    key: "disease"
    value {
        description: "A disease"

        features {
            key: "disease_id"
            value: {
                description: "Disease ID originating from OMIM"
                dtype: DT_STRING
                shape { dim { size: -1 } }
            }
        }

        features {
            key: "disease_name"
            value: {
                description: "Disease name as a bag of words, like Lynch syndrome"
                dtype: DT_STRING
                shape { dim { size: -1 } }
            }
        }
    }
}

node_sets {
    key: "variant"
    value {
        description: "A gene variant"

        features {
            key: "label"
            value : {
                description: "Ground truth label, is the causative variant for the case"
                dtype: DT_FLOAT
            }
        }

        features {
            key: "rank_score"
            value: {
                description: "The combined inferred pathogenicity score from RD pipeline"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "gene-variant"
    value {
        description: "A gene-to-variant link"
        source: "gene"
        target: "variant"

        features {
            key: "relatedness"
            value: {
                description: "A boolean relationship, [0, 1]"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "gene-disease"
    value {
        description: "A gene-to-disease link"
        source: "gene"
        target: "disease"

        features {
            key: "relatedness"
            value: {
                description: "A boolean relationship, [0, 1]"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "hpo-hpo"
    value {
        description: "HPO-HPO ontology link"
        source: "hpo"
        target: "hpo"

        features {
            key: "relatedness"
            value: {
                description: "A boolean relationship, [0, 1]"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "hpo-gene"
    value {
        description: "HPO-to-gene link"
        source: "hpo"
        target: "gene"

        features {
            key: "relatedness"
            value: {
                description: "A boolean relationship, [0, 1]"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "disease-to-hpo"
    value {
        description: "Disease-to-HPO link"
        source: "disease"
        target: "hpo"

        features {
            key: "frequency"
            value: {
                description: "Frequency of HPO occurrence in Disease"
                dtype: DT_FLOAT
            }
        }
    }
}

"""