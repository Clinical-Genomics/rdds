from tempfile import NamedTemporaryFile
import tensorflow as tf

from rdds.phen2gen.dataset.sampler import InMemorySampler
from rdds.phen2gen.dataset.dataset import Phen2GenDatasetCompiler

def test_sampler():
    tmpfile = NamedTemporaryFile(dir='/tmp')
    compiler = Phen2GenDatasetCompiler(tfrecord_output_path=tmpfile.name)
    graph_iterator = compiler._yield_graph_tensor(dummy_data=True)
    single_graph = graph_iterator.__next__()
    schema = compiler.schema
    sampler = InMemorySampler(graph_schema=schema,
                              complete_graph=single_graph)
    sampling_model = sampler.build_sampling_model()

    seeds = tf.data.Dataset.from_tensor_slices([0, 1])
    # Create batches of up to two seeds
    seeds = seeds.batch(1)
    # [seed1, seed2] -> [[seed1], [seed2]]
    seeds = seeds.map(
        lambda s: tf.RaggedTensor.from_row_lengths(s, tf.ones_like(s))
    )
    graphs = seeds.map(sampler.samping_model)
    for graph in graphs:
        print(graph)