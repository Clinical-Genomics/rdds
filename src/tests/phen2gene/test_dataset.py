from tempfile import NamedTemporaryFile
from rdds.phen2gen.dataset import Phen2GenDatasetCompiler

def test_create_dataset():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    dataset_compiler._write_tfrecord(dummy_data=True)


def test_compile_intermediate_graph_dataset():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    dataset_compiler.compile_graph_blob()

def test_compile_variant_nodes():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path='/rdds/tmp/justhusky_short.vcf')

def test_compile_vcf_to_intermediate_graph():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    intermediate_graph = dataset_compiler._load_intermediate_graph()
    variant_nodes = dataset_compiler._construct_variant_nodes(vcf_path='/rdds/tmp/justhusky_short.vcf')

