# This file contains the data schema definition
# https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/schema.md

from tensorflow_gnn import NODES, EDGES

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
            key: "hpo_id_full"
            value: {
                description: "ID of the HPO term, like HP:0030992"
                dtype: DT_STRING
            }
        }

        features {
            key: "hpo_name"
            value: {
                description: "HPO human readable name as bag of words, like Abnormal pancreatic duct morphology"
                dtype: DT_STRING
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
            }
        }

        features {
            key: "disease_name"
            value: {
                description: "Disease name as a bag of words, like Lynch syndrome"
                dtype: DT_STRING
            }
        }
    }
}

node_sets {
    key: "variant"
    value {
        description: "A gene variant"

        features {
            key: "variant_id"
            value: {
                description: "Variant ID"
                dtype: DT_STRING
            }
        }

        features {
            key: "genmod_rank_score"
            value: {
                description: "Inferred pathogenicity score from the RD pipeline"
                dtype: DT_FLOAT
            }
        }

        features {
            key: "label"
            value : {
                description: "Ground truth label, is the causative variant for the case"
                dtype: DT_FLOAT
            }
        }
    }
}

edge_sets {
    key: "gene>variant"
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
    key: "gene>disease"
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
    key: "hpo>hpo"
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
    key: "hpo>gene"
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
    key: "hpo>disease"
    value {
        description: "Disease-to-HPO link"
        source: "hpo"
        target: "disease"

        features {
            key: "frequency"
            value: {
                description: "Frequency of HPO occurrence in Disease"
                dtype: DT_FLOAT
            }
        }
    }
}

node_sets {
    key: "latent-variant-hpo"
    value {
        description: "A latent node representing a variant - HPO association"
    }
}

edge_sets {
    key: "variant>latent-variant-hpo"
    value {
        description: "Variant to latent"
        source: "variant"
        target: "latent-variant-hpo"
    }
}

edge_sets {
    key: "hpo>latent-variant-hpo"
    value {
        description: "HPO to latent"
        source: "hpo"
        target: "latent-variant-hpo"
    }
}
"""
# TODO: Add "hypothesis" feature to possible seed nodes?
# TODO: Decide whether to use latent node edges as the "strength" between variant-hpo or use the latent state itself
# TODO: Add latent node features "hypothesis" as prediction y_hat?

_DUMMY_DATA = {
    (NODES, "hpo", "hpo_id"): [0, 1],
    (NODES, "hpo", "hpo_name"): ["hpo_name0", "hpo_name1"],
    (NODES, "gene", "gene_id"): [2, 3],
    (NODES, "gene", "gene_symbol"):["BRCA1", "BRCA2"],
    (NODES, "disease", "disease_id"): ["OMIM:0", "OMIM:1"],
    (NODES, "disease", "disease_name"): ["Developmental and epileptic encephalopathy", "Short QT syndrome 2"],
    (NODES, "variant", "variant_id"): ["1.3213A>T", "5.5421235G>C"],
    (NODES, "variant", "rank_score"): [0.01, 0.95],
    (NODES, "variant", "label"): [0, 1],
    (EDGES, "gene>variant", "relatedness"): [0, 1],
    (EDGES, "gene>disease", "relatedness"): [0, 1],
    (EDGES, "hpo>hpo", "relatedness"): [0, 1],
    (EDGES, "hpo>gene", "relatedness"): [0, 1],
    (EDGES, "disease>hpo", "relatedness"): [0, 1]
}