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
-> Treat this as a link-prediction problem.

I think one can see the `node state` in tf-gnn as the node embedding vector.
The goal is then to reduce distance between linked nodes, using the `node states` as embeddings.

> Possibly usecase for contrastive learning (SimCLR, SimSiam, and MOCO) to infer links (node states; embeddings) ?

### Data

For training data, use HPO and metadata. Use all of the HPO terms for training.
For evaluation use case HPO terms and a gene panel.
This is a bit awkward, since we want to use all of the existing knowledge to
learn node embeddings.

Another option is to use all of HPO data, and use CG case variants (and associated gene)
to test model performance (as a validation set).

> Use only Orphanet annotations (ignoring OMIM)

[Great list of additional data sources by OMIM](https://mirror.omim.org/help/external)

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

> Design Obstacle: Variants are case-specific and not available at time of generation. Need to have
    an embedding model that 1. does not use variants for training graphSAGE or 2. is somehow agnostic to variants present or not.
    Or we do this fully explorative, that is, train SAGE and infer links on a case-by-case basis. THIS IS THE STARTING POINT!
    Can I decompose the SAGE kernel somehow, to allow injection of variant based weights later on?
    Generating embeddings REQUIRES that all data types are present at time of latent variable creation!

### Integrations
- [ ] Decide on whether to generate a gene panel, or to use as per-variant scoring method

## APIs
- [Tensorflow-GNN](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/overview.md)
    - [Graph Schema docs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/schema.md) and associated schema validation tools
    - [... and how to instantiate a Schema using GraphTensor](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/graph_tensor.md)
    - [Helpful example on how to read data from disk using Schema and GraphTensor proto files](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md#file-input-and-parsing)
    - [Example on defining schema in python kernel and verifying it](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/log_2022/neurips_teacher_tfgnn_graph_classification_mutag.ipynb)
    - [Pseudocode on full process, including pipeline and model training](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md#the-big-picture-training-export-and-inference) 
    - [On seed nodes, graph readout structures for training](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/schema.md#about-labels-and-reading-out-the-final-gnn-states) 
    - [On how to adapt Keras layers to deal with tfgnn outermost dimension](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md#the-shape-of-features)
- [TF-GNN: Graph Neural Networks in TensorFlow, Arxiv](https://arxiv.org/pdf/2207.03522)
- [Graph Neural Networks in TensorFlow: A Practical Guide](https://drive.google.com/file/d/1zn6qIPnwFktYUsTbjRQVkO5n0TmlewMR/view)
    Also contains interesting links to examples where the GNN learns the graph relations itself in unsupervised fashion (UGSL).
    Great slides for different applications and domains, visualised.
- Check out [Grale: Designing Networks for Graph Learning, Jonathan Halcrow, Alexandru Mosoi, Sam Ruth, Bryan Perozzi]()
- [Visualizing GNNs with NetworkX](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/kdd_2023/code_tutorial_visualization.ipynb)
- [Visualizing very large networkx graphs using Datashader](https://datashader.org/user_guide/Networks.html)
- [Additional visualisation example in Keras tutorials, GNN for node classification](https://keras.io/examples/graph/gnn_citations/)
- [Important notes on pattern for Model Saving and Inference](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/input_pipeline.md#the-big-picture-training-export-and-inference)
- [Tensorflow GNN node classification](https://github.com/tensorflow/gnn/blob/main/examples/tutorials/log_2022/code_tutorial_1_tfgnn_single_machine.ipynb)
- [Contrastive losses API for Self-supervised learning, SSL](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/api_docs/python/models/contrastive_losses.md)

## Related Source Code
- [Graph2Vec in Keras and NetworkX](https://keras.io/examples/graph/node2vec_movielens/)
    Word2Vec does not account for the node features! It's up to developer to design the network graph before running graph2vec.
    Furthermore, word2vec does not provide a method to do inductive embeddings (i.e. embeddings for inference) without
    having access to the whole graph. This book also discusses MPNN in detail for generating embeddings.

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
8. [SCOUT Phenotype support](https://github.com/Clinical-Genomics/scout/blob/ecb0ad3288f09fad11ffd8f76c85a4e6f0ae7fde/docs/features/hpo.md)

- [Book on Graph Learning](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)
    Great for terminology and defining scope for ML problems. Good explanations of concepts.

- [Link prediction, ScienceDirect](https://www.sciencedirect.com/topics/computer-science/link-prediction)

# Alternative take on Case Investigation

The overall goal is to find matching variants, given some clues about the patient.
Clues like phenotype, expected genes (gene panel) etc

Can I use information on benign variants (low scoring) to infer labels as well?

Can I generate a latent node which is a guess vector?

Can I infer links to HPO and Disease nodes?

- Build GNN using variant data
- Enrich data using HPO terms
- Use solved case causative variant to try to identify related gene using HPO terms or other mining technique

GOAL:
The hope is that by filling in some SMALL portion of the overall graph (links) it's possible to connect
large portions of the graphs together associating variants with disease.

Method:
1. Create graph from known relationships, like HPO and genes, disease
2. Either infer a new graph using variant information (like gene annotations etc)
3. or create embeddings to (construct a similarity graph?) Embeddings only useful if we want a fixed dimension as output.

Perhaps this has to be done in a two tier fashion:
1. Create graph from existing information 
2. ?

FOR EVERY NODE, IDENTIFY WETHER IT'S PATHOGENIC OR NOT
PROPAGATE THE PATHOGENIC INFORMATION TO NEIGHBOURING NODES

For example, assuming some set of HPO terms, diagnosis from user is marked as causative by clinician (seed nodes)
as well as a set of genes (by gene panel). What variants will be marked as causative by association of the
causative label is propagated by linking?
Variants with higher conductance to causative non-variant seed nodes should naturally be considered more pathogenic.
One can refine the variance initial causativity/ state/ label by MIVMIR score for example.
This can then also be applied to Disease nodes for disease recommendation or for differentiation.

Label propagation should not consider variant-variant conductance? No not in the first place.
Or perhaps we should, in case we add in CLINVAR variants (it then needs to propagate on CHROM.POS) alongside
patient variants.

Need to formulate label propagation as a function of conductance between nodes, that's specific to the node type.

One can also view this as a network association problem, to identify overlapping networks (hpo-disease-variant, gene-variant).

If we successfully manage to cluster and graph the data by data features and existing links, it should improve label conductance
for the causative variant. HOW DO I DEFINE A FUNCTION THAT INCREASE CONDUCTANCE BETWEEN INITIAL CLINICAL GUESS AND
THE CAUSATIVE VARIANT.

Method:
1. Create relational graph from known existing links
2. Infer additional links using node features (optimize conductance for causative variant to clinical seed information)
3. Propagate clinical causative guess from seed node to variants
4. Pick out most likely causative variant and disease
5. Additionally filter on disease by clinician to narrow down the variants

In case I decide to train message passing for conductance learning, I need to vary the inputs to the seeds.
The seed nodes will be the input x to predict y_hat. One can vary the seed nodes by:
- gene
- hpo terms (dropout)

[Message passing example in this TFGNN arxiv paper](https://arxiv.org/pdf/2207.03522)
[Message passing paper for chemistry to embed molecules, arxiv](https://arxiv.org/pdf/1704.01212) but it read out on all of graph,
not individual nodes.

In case of no conductance learning (i.e. graph building using for example GRALE)
there's no dependency on the inputs. We rather evaluate the graph quality
by precision, recall on the known causative variant.

## As a Node Prediction Problem
Use [GraphSAGE](https://arxiv.org/pdf/1706.02216) can be used to generate unsupervised embeddings from
node-feature-neighbor data.
[Description of GraphSage](https://snap.stanford.edu/graphsage/) and [GraphSAGE imlementation in Tensorflow](https://github.com/williamleif/GraphSAGE)
Then we frame it as a node classification problem.

# Google GNN introduction NEURIPS

Recognize two types of graph models:
- Relationational graph (external source)
- Similarity graph (embeddings)

Watch: https://gm-neurips-2020.github.io/

[Graph Mining tools for arbitrary data to build similarity graph using Grale](https://www.youtube.com/watch?v=l0j2oscDKRA)
[Grap similarity metrics to infer links, embeddings and do clustering](https://www.youtube.com/watch?v=30vevrzV-cM)
Reinforcement learning to try to learn a function to infer a graph model from:
- relationships (of varying quality)
and to use the graph to infer
- variant pathogenicity

Problem statement: learn a function to associate causative VARIANT <-> Set of HPOs
Use inference time HPOs to pick out neighbouring clustered candidate causative variants that's associated with the HPO
terms in association with it's neighbouring data points.

Look into GRALE for graph learning, to create a graph from data.

Look into Node2Vec for graph similarity tasks.
https://snap.stanford.edu/node2vec/
[Node2Vec implementation](https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py)
[Node2Vec in TF](https://github.com/apple2373/node2vec)

"Affinity Hierarchichial Clustering" algorith for graph clustering seems to outperform other algos like k-means.
This algo tries to reduce interconnections across identified clusters (intercluster edges).
Applied to graphs.

"Metric Clustering" tries to cluster a set of points (embeddings).
Tries to minimize k-center, k-means and k-median losses.

The way above two clsutering techniques are solved in a distributed fashion, is to chunk the data
using LSH chunking (project data on axis and bin the data) and run clustering algo on each LSH bin data subset.

"Coverage maximisation" is a technique to maximize cluster overlap with neighbouring datapioints.

"Community detection" tries to identify small clusters of datapoints,  sparsley cluster-interconnected.
Very fine grained clustering. Related clustering metrics are `modularity`, `conductance`, `density`.
Algos often used are `spectral`, `MCL`, `Infomap` or from google `cocondunctance`.

Semi-supervised Learning [SSL](https://www.youtube.com/watch?v=A6dBO64zwq4). Using `seed nodes` annotated with ground truth labels,
which is then propagated to similar nodes by edges, expanding the annotated labels.
A graph edge can be a similarity metric of any type.
The labels can actually be any type of data or information thats propagated to neighbour.
By holding out a portion of the truth labels for validation, one can compute precision recall
on the learned label set. Apply thresholding to pick out "strong" learned labels.
Inputs to SSL is similarity signal (weighted edges), training and test labels.
Output is learned labels for nodes.
One can also have a model to incorporate features as part of the label update step,
incorporating neigbour labels and node features.

SSL applications:
- Multi class classification automagically
- Label cleaning by adjusting incorrect, noisy labels
- NLP sentiment and emotion detection by identifying synynomous phrases (can be applied to variants?)
Can we use SLL to iteratively identify datapoints of interest, given a set of starting points?

## GNNs intro by Google GM NeurIPS
[1. GNNs and graph embeddings](https://www.youtube.com/watch?v=sgRY9-p7z20)
Look into article `Machine Leaning on Graphs: A Model and Comprehensive Taxonomy`, arXiv

GCNs fail due to over-averaging (smooting) in convolutional layers that's aggregating node neighborhoods.
For example, where a node has the opposite label as its neighbors. MixHop algorithm seems to overcome this
averaging issue in GCN convolutional layers.

GCNs have difficulty deciding what neighbor nodes are important, hence the Personalized Page Rank (PPR)[https://www.youtube.com/watch?v=J0m4NnTnft8]
algorithm. It basically keeps track of how often a neighboring node appears in a n random walk experiment,
assigning high weight to commonly appearing nodes.

Debiasing GNNs can be done using [MONET algorith](MONET: debiasing graph embeddings via the metadata-orthogonal training unit, arxiv).
This removes metadata leakage from the graph embedding step (for a select set of metadata) so that
it does not affect the embeddings.

Generating graph embeddings
- Deepwalk algorithm (treat node IDs sequence in random walk as a NLP sentence encoding problem)
- EgoNetwork algorithm disentangles sub graph networks that would otherwise be mistakenly associated in a vanilla random
    walk embedding. This is done by selecting the ego node (seed) and remove it from the subgraph network,
    to generate (potentially) disjoint set of graphs in case the ego node was the sole connection between them.
    EgoNets are extended into [`Persona Graphs`](https://www.youtube.com/watch?v=5ZRZYePjS0c) which is basically creates an new graph (and later embedding) of each disjoint, disconnected
    subgraph (containing a replica of the Ego node). The `Persona Graph` embeddings are used as an additional dimension
    to the default graph embeddings (now representing subgraphs). Especially useful in `Link Prediction` problems.

- Great link prediction example: https://github.com/tensorflow/gnn/blob/main/examples/tutorials/kdd_2023/code_tutorial_1.ipynb
- Great intro to GNNs: https://gnn.seas.upenn.edu/lectures/lecture-1/