"""
Sampling specification for graph sampler.
Only nodes and edges listed in this specification will be sampled during training.
"""

_SAMPLING_SPEC = \
"""
seed_op {
  op_name: "SEED->latent-variant-hpo"
  node_set_name: "latent-variant-hpo"
}
sampling_ops {
  op_name: "variant>latent-variant-hpo"
  input_op_names: "SEED->latent-variant-hpo"
  edge_set_name: "variant>latent-variant-hpo"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
sampling_ops {
  op_name: "hpo>latent-variant-hpo"
  input_op_names: "SEED->latent-variant-hpo"
  edge_set_name: "hpo>latent-variant-hpo"
  sample_size: 10
  strategy: RANDOM_UNIFORM
}
"""