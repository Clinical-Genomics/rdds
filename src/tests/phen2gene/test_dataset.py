from tempfile import NamedTemporaryFile
import tensorflow_gnn as tfgnn

from rdds.phen2gen.dataset import Phen2GenDatasetCompiler

def test_create_dataset():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    dataset_compiler._write_tfrecord(dummy_data=True)


def test_compile_intermediate_graph_dataset():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    dataset_compiler.compile_graph_blob()

def test_compile_variant_nodes_edges():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    intermediate_graph = dataset_compiler._load_intermediate_graph()
    vcf_path = '/rdds/tmp/justhusky_short.vcf'
    variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path=vcf_path)
    variant_gene_edges = dataset_compiler._construct_variant_gene_edges(vcf_path=vcf_path,
                                                              variant_nodes=variant_nodes,
                                                              gene_nodes=intermediate_graph.gene_nodes)

def test_compile_vcf_to_intermediate_graph():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    intermediate_graph = dataset_compiler._load_intermediate_graph()
    vcf_path = '/rdds/tmp/justhusky_short.vcf'
    variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path=vcf_path)
    variant_gene_edges = dataset_compiler._construct_variant_gene_edges(vcf_path=vcf_path,
                                                              variant_nodes=variant_nodes,
                                                              gene_nodes=intermediate_graph.gene_nodes)

    graph = dataset_compiler._build_graph(intermediate_graph=intermediate_graph,
                                          variant_nodes=variant_nodes,
                                          variant_gene_edges=variant_gene_edges)
    assert isinstance(graph, tfgnn.GraphTensor)


