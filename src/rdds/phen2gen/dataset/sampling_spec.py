"""
Sampling specification for graph sampler.
Only nodes and edges listed in this specification will be sampled during training.
"""

_SAMPLING_SPEC = \
"""
seed_op {
  op_name: "seed"
  node_set_name: "hpo"
}
sampling_ops {
  op_name: "hpo>gene"
  input_op_names: "seed"
  edge_set_name: "hpo>gene"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "variant>gene"
  input_op_names: "hpo>gene"
  edge_set_name: "variant>gene"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
"""