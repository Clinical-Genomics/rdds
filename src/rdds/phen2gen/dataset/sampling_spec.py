"""
Sampling specification for graph sampler.
Only nodes and edges listed in this specification will be sampled during training.
"""

_SAMPLING_SPEC = \
"""
seed_op {
  op_name: "seed"
  node_set_name: "variant"
}
sampling_ops {
  op_name: "variant-gene"
  input_op_names: "seed"
  edge_set_name: "variant>gene"
  sample_size: 1
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "gene-hpo"
  input_op_names: "variant-gene"
  edge_set_name: "gene>hpo"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "hpo-hpo"
  input_op_names: "gene-hpo"
  edge_set_name: "hpo>hpo"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
"""