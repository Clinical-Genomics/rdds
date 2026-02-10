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

    """
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
        """

    def _build_sampling_spec(self):
        self._sampling_spec = text_format.Parse(_SAMPLING_SPEC, tfgnn.sampler.SamplingSpec())

    def _build_sampling_model(self):
        """
        TFGNN sampling model expects input as RaggedTensor with shapes [batch_size (components), seed indexes]
        Example: [[1], [1412]], n_components=2 with 1 seed node per component (batch)
        """

        # TODO: Set seeds in sampling from_graph_tensor()

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
        assert self._sampling_spec
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
        _LOGGER.info(f"Sampling spec:\n{self._sampling_spec}")
        self._build_sampling_model()

    @property
    def amount_seed_nodes(self) -> int:
        """
        Get the sampling seed node set name from sampling spec;
        seed_op {
            op_name: "seed"
            node_set_name: "nodeName"
        }
        and return amount of seed nodes in GraphTensor.
        """
        seed_node_name: str = None
        msg = text_format.MessageToString(self._sampling_spec)
        msg = msg.split('}')  # Split on sampling sub specs
        msg = [m.split('\n') for m in msg]
        for parts in msg:
            for part in parts:
                if 'seed_op' in part:
                    # Now in the seed_op definition, find the node_set_name
                    for seed_op_part in parts:
                        if 'node_set_name' in seed_op_part:
                            seed_node_name = seed_op_part.replace(' ', '').replace('"','').split('node_set_name:')[1]
                            break
        assert seed_node_name
        size_tensor = self._complete_graph.node_sets[seed_node_name].total_size.numpy()
        n_seed_nodes = int(size_tensor)  # Test for casting data to 1D int
        return n_seed_nodes