from tensorflow import int32, int64
from tensorflow.keras import Model as KerasModel
import tensorflow_gnn as tfgnn
from tensorflow_gnn.experimental import sampler as expsampler
from tensorflow_gnn.sampler import sampling_spec_pb2
from google.protobuf import text_format


from .. import _LOGGER
from .sampling_spec import _SAMPLING_SPEC

# TODO: Consider sample size as hyperparameters

class InMemorySampler:

    def __init__(self,
                 graph_schema: tfgnn.GraphSchema,
                 complete_graph: tfgnn.GraphTensor):
        self._graph_schema = graph_schema
        self._complete_graph = complete_graph

    def _build_sampling_spec_broken(self):
        sampling_spec_builder = tfgnn.sampler.SamplingSpecBuilder(
            self._graph_schema,
            default_strategy=tfgnn.sampler.SamplingStrategy.RANDOM_UNIFORM)
        # Define sampling seed nodes
        sampling_spec_builder = sampling_spec_builder.seed('latent-variant-hpo')
        # ... and what context to sample
        sample_op_variants = sampling_spec_builder.sample(sample_size=10,  # Sample 10 variants maximum
                                                          edge_set_name='variant>latent-variant-hpo')
        sample_op_hpos = sampling_spec_builder.sample(sample_size=10, # Sample 10 HPOs maximum
                                                      edge_set_name='hpo>latent-variant-hpo')
        # Sample
        # merge_then_sample ?
        sampling_spec_builder = sample_op_variants.join([sample_op_hpos]).sample()
        #sampling_spec_builder = sample_op_hpos.merge_then_sample(other_steps=[sample_op_variants])
        # FIXME: Sampling here is broken atm, difficult to join DAG sampling ops

        # Try alternative sampling technique
        sampling_spec_builder = tfgnn.sampler.SamplingSpecBuilder(
            self._graph_schema,
            default_strategy=tfgnn.sampler.SamplingStrategy.RANDOM_UNIFORM)
        sampling_spec_builder = sampling_spec_builder.seed('latent-variant-hpo')
        sampling_spec_builder = sampling_spec_builder.sample(sample_size=10,  # Sample 10 variants maximum
                                                          edge_set_name='variant>latent-variant-hpo')
        sampling_spec_builder = sampling_spec_builder.sample(sample_size=10,  # Sample 10 HPOs maximum
                                                      edge_set_name='hpo>latent-variant-hpo')
        sampling_spec = sampling_spec_builder.build()


        self._sampling_spec: tfgnn.sampler.SamplingSpec = (sampling_spec_builder).build()
        print(self._sampling_spec)

    def _build_sampling_spec(self):
        self._sampling_spec = text_format.Parse(_SAMPLING_SPEC, tfgnn.sampler.SamplingSpec())

    def _build_sampling_model(self):

        def edge_sampler(sampling_op: tfgnn.sampler.SamplingOp):
            edge_set_name = sampling_op.edge_set_name
            sample_size = sampling_op.sample_size
            return expsampler.InMemUniformEdgesSampler.from_graph_tensor(
                self._complete_graph, edge_set_name, sample_size=sample_size, name=sampling_op.op_name
            )

        def get_features(node_set_name: tfgnn.NodeSetName):
            return expsampler.InMemIndexToFeaturesAccessor.from_graph_tensor(
                self._complete_graph, node_set_name
            )

        # NOTE: seed_node_dtype must go hand in hand with tfgnn.create_graph_spec_from_schema_pb(..., indices_dtype=tf.int64)
        self._sampling_model: KerasModel = expsampler.create_sampling_model_from_spec(
            self._graph_schema,
            self._sampling_spec,
            edge_sampler,
            get_features,
            seed_node_dtype=int64)
        _LOGGER.info(f"Sampling model:\n{self._sampling_model.summary(line_length=120)}")

    @property
    def samping_model(self) -> KerasModel:
        return self._sampling_model

    def build_sampling_model(self):
        self._build_sampling_spec()
        self._build_sampling_model()