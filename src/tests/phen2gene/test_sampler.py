from tempfile import NamedTemporaryFile
import tensorflow as tf
from tensorflow_gnn.keras.layers.graph_ops import GraphTensor

from rdds.phen2gen.dataset.sampler import InMemorySampler
from rdds.phen2gen.dataset.dataset import Phen2GenDatasetCompiler

def test_sampler():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    graph_iterator = compiler._yield_graph_tensor(dummy_data=True)
    single_graph = graph_iterator.__next__()  # The graph that's sampled from
    schema = compiler.schema
    sampler = InMemorySampler(graph_schema=schema,
                              complete_graph=single_graph)
    sampler.build_sampling_model()

    seeds = [0, 1]
    n_seeds = len(seeds)
    seeds = tf.data.Dataset.from_tensor_slices(seeds)
    seeds = seeds.batch(1)  # n_components in GraphTensor
    seeds = seeds.map(
        lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
    )
    graphs = seeds.map(sampler.samping_model)
    count = 0
    for graph in graphs:
        assert isinstance(graph, GraphTensor)
        count += 1
    assert count == n_seeds

def test_sampler_with_real_data():
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
