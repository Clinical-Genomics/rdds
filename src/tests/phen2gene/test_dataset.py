from tempfile import NamedTemporaryFile
from rdds.phen2gen.dataset import Phen2GenDatasetCompiler

def test_create_dataset():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    dataset_compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    dataset_compiler._write_tfrecord(dummy_data=True)