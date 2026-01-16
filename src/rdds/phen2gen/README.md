# Phen2Gen

Phenotype to Genotype inference module.

## Design and Implementation Strategy

Do as in 2. to encode phenotype - gene relations using word2vec.
A simpler approach (not using patient case data) would be to go with 3. instead.

In a later step, add in ortholog gene data and additional attributes:
- [ ] Frequency of a clinical feature (HPO term) within a disease (from HPO)

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