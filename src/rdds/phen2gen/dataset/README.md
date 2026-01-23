# Dataset

In short, this module takes various input data sources (files on disk) and creates serialized
`tf.Example` files based on sampled `tfgnn.Graph` instances.

https://github.com/tensorflow/gnn/blob/main/tensorflow_gnn/docs/guide/data_prep.md

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
