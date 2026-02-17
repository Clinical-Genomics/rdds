# Dataset

In short, this module takes various input data sources (files on disk) and creates serialized
`tf.Example` files based on sampled `tfgnn.Graph` instances.

https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md

- [ ] TODO: Design decision: To represent the whole case in a GraphTensor or to do graph sampling
  - Pros complete graph: All relevant features available immediately, long term data
  - Cons complete graph: Risk of overfitting when all data available? Slower convergence. Poor scalability
  - [Sampling methods for efficient training of graph convolutional networks: A survey](https://arxiv.org/pdf/2103.05872)

[Good examples on sampling configs](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/sampler/sampling_spec_builder_test.py)

This is my usecase: https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/sampler/sampling_spec_builder_test.py#L136

## SamplingSpecBuilder
The python API enforces source->target edge directionality in sampling which makes the python API not as
flexible as the protobuf definition which allows bidirectional sampling.

## Sampler

```
Sampling is meant to give the seed node a sufficiently large neighborhood
such that a GNN model can compute a useful hidden state.
```
[src](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md#sampler-configuration)

There are two options to do graph sampling:
1. InMemory sampling (all data in RAM, in `GraphTensor` instance) to create `tf.Example` records on disk
2. Sampling from a `unigraph`-file to `tf.Example` and `GraphSpec` using the [Beam sampler](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/beam_sampler.md).

### InMemory Sampling
- [InRAM sampling](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/inmemory_sampler.md)

### Beam Sampler
- [Graph sampling for large networks not fitting in RAM](https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md#graph-sampling)
Input to the `BeamSampler` is a `unigraph` proto format file that contains the complete graph.

Sampler takes three inputs:
- Unigraph file
- Sampling specification
- Seed nodes to sample (optional) which selects the origo of sampling for neighborhood gathering

The output is serialized `tf.Example` proto files containing serialized `GraphTensor`s.
