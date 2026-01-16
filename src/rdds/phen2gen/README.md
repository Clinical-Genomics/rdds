# Phen2Gen

Phenotype to Genotype inference module.

The goal of this module is to infer related gene(s) for a set of HPO terms.

Basic principle is that for a given set of HPO terms,`H`<sub>n</sub>,
learn (and later infer) related genes `G`<sub>m</sub>.
As `n` increases, it's likely that `m` will decrease and precision will increase. 

## Design and Implementation Strategy

Let HPO terms and Genes be represented as two separate type sets of nodes, `H` and `G`.

`H` links not only to itself (by the HPO onthology) but also to `G`.
`H` and `G` are linked using `a-is` link from the HPO onthology.

In addition, let Disease be a third set of nodes `D` that links `G` and `H`.

For a given subset of `H`, learn to infer a link strength (0, 1) from `H -> G` using the training labels.
using `G` embedding and `D` embeddings based on HPO frequencies.

Question: Do we want to create embeddings of the nodes H, G and D OR NOT?
Do we want to apply a Convolutional-GNN (based on embeddings) or do some kind of message-passing schema?
Seems like creating node embeddings is the way to go, based on 1. and 3.
See [chapter 12.3.2 (Inductive Knowledge Graph Embeddings) and 12.4 (Recommender Systems)](file:///tmp/978-3-031-01587-8.pdf)

I think one can see the `node state` in tf-gnn as the node embedding vector.
The goal is then to reduce distance between linked nodes, using the `node states` as embeddings.

### Data

For training data, use HPO and metadata. Use all of the HPO terms for training.
For evaluation use case HPO terms and a gene panel.
This is a bit awkward, since we want to use all of the existing knowledge to
learn node embeddings.

Another option is to use all of HPO data, and use CG case variants (and associated gene)
to test model performance (as a validation set).

> Use only Orphanet annotations (ignoring OMIM)

### Model
Do as in 2. to encode phenotype - gene relations using word2vec.
A simpler approach (not using patient case data) would be to go with 3. instead.

In a later step, add in ortholog gene data and additional attributes:
- [ ] [Frequency of a clinical feature](https://obophenotype.github.io/human-phenotype-ontology/annotations/frequency/) (HPO term) within a disease (from HPO)
  - Which is dependent on [genes to disease](https://obophenotype.github.io/human-phenotype-ontology/annotations/genes_to_disease/) data
- ClinVar metadata?

Focus on the HPO-Orphanet ? Or should we use HPO-OMIM instead? Is there a difference w.r.t what clinicians use for
interpretation? OMIM is the standard in clinic.
However, using Orphanet annotations is better from a clinical perspective <sup>3.</sup>

### Training Target for Generating Embeddings
1. Predict pheno-pheno relationships
2. Predict pheno-geno relationships ?
3. Predict pheno-[META] relationship?

> Can I train the embeddings using a two-goal method: to correctly identify pheno-pheno and pheno-gene links?
Can this be extended to other properties, like to infer pheno-mouse-gene to improve embeddings?

## APIs
- [Tensorflow-GNN](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md)

## References
1. [PhenoLinker: Phenotype-Gene Link Prediction and Explanation using Heterogeneous Graph Neural Networks](https://arxiv.org/html/2402.01809v1)
2. [CADA: phenotype-driven gene prioritization based on a case-enriched knowledge graph](https://pmc.ncbi.nlm.nih.gov/articles/PMC8415429/#B16)
3. [HPO2Vec+: Leveraging Heterogeneous Knowledge Resources to Enrich Node Embeddings for the Human Phenotype Ontology](https://pmc.ncbi.nlm.nih.gov/articles/PMC6710011/)
4. [Graph-Based Link Prediction between Human Phenotypes and Genes](https://onlinelibrary.wiley.com/doi/10.1155/2022/7111647)
5. [Heterogeneous Information Network Embedding for Recommendation](https://arxiv.org/pdf/1711.10730)
6. [Interpretable Clinical Genomics with a Likelihood Ratio Paradigm](https://pmc.ncbi.nlm.nih.gov/articles/PMC7477017/)
Good methods for testing robustness.
7. [PhenoDigm: analyzing curated annotations to associate animal models with human diseases](https://academic.oup.com/database/article/doi/10.1093/database/bat025/333089)
Good list of ortholog (gene) data sources