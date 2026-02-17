from tempfile import NamedTemporaryFile
import tensorflow as tf
from tensorflow_gnn import GraphTensor, enable_graph_tensor_validation_at_runtime

from rdds.phen2gen.dataset.sampler import InMemorySampler
from rdds.phen2gen.dataset.dataset import Phen2GenDatasetCompiler
from rdds.phen2gen.dataset.schema import _DUMMY_DATA
from rdds.lib.tfgnn import random_graph_tensor_with_id
import pytest as pt

_SINGLE_GRAPH = (1, 2)  # [1, 2)


def test_sampler_with_dummy_data():
    enable_graph_tensor_validation_at_runtime()
    tf.keras.backend.clear_session()
    tmpfile = NamedTemporaryFile(dir='/tmp')
    compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    # NOTE: Be aware of limitation in random_graph_tensor(), cannot generate graph_tensor.indices_dtype int64
    graph_tensor = random_graph_tensor_with_id(spec=compiler.graph_spec,
                                               sample_dict=_DUMMY_DATA,
                                               num_components_range=_SINGLE_GRAPH,
                                               validate=True)
    sampler = InMemorySampler(graph_schema=compiler.schema,
                              complete_graph=graph_tensor)
    sampler.build_sampling_model()
    n_seeds = sampler.amount_seed_nodes
    seeds = list(range(0, n_seeds))
    seeds = tf.data.Dataset.from_tensor_slices(seeds)
    seeds = seeds.batch(1)  # n_components in GraphTensor
    seeds = seeds.map(
        # sampling model expects RaggedTensor with shapes [batch_size (components), seed indexes]
        lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
    )
    graphs = seeds.map(sampler.samping_model)
    graphs = graphs.prefetch(n_seeds)
    count = 0
    for graph in graphs:
        assert isinstance(graph, GraphTensor)
        count += graph.total_num_components.numpy()  # Sum batch dimension
    assert count == n_seeds

@pt.mark.parametrize("n_iter", [1, 5])
def test_repeat_dummy_data(n_iter):
    for _ in range(0, n_iter):
        test_sampler_with_dummy_data()

def test_sampler_with_real_data():
    enable_graph_tensor_validation_at_runtime()
    tf.keras.backend.clear_session()
    vcf_path = '/rdds/tmp/justhusky_short.vcf'
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=None)
    schema = dataset_compiler.schema
    intermediate_graph = dataset_compiler._load_intermediate_graph()
    variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path=vcf_path)
    variant_gene_edges = dataset_compiler._construct_variant_gene_edges(vcf_path=vcf_path,
                                                                        variant_nodes=variant_nodes,
                                                                        gene_nodes=intermediate_graph.gene_nodes)

    graph = dataset_compiler._build_graph(intermediate_graph=intermediate_graph,
                                          variant_nodes=variant_nodes,
                                          variant_gene_edges=variant_gene_edges)
    sampler = InMemorySampler(graph_schema=schema,
                              complete_graph=graph)
    sampler.build_sampling_model()

    n_seeds = sampler.amount_seed_nodes
    seeds = list(range(0, n_seeds))
    seeds = tf.data.Dataset.from_tensor_slices(seeds)
    seeds = seeds.batch(1024)  # n_components in GraphTensor
    seeds = seeds.map(
        # sampling model expects RaggedTensor with shapes [batch_size (components), seed indexes]
        lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
    )
    graphs = seeds.map(sampler.samping_model)
    graphs = graphs.prefetch(n_seeds)
    count = 0
    for graph in graphs:
        assert isinstance(graph, GraphTensor)
        count += graph.total_num_components.numpy()
    assert count == n_seeds

@pt.mark.parametrize("n_iter", [1, 5])
def test_repeat_real_data(n_iter):
    for _ in range(0, n_iter):
        test_sampler_with_real_data()