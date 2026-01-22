import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tempfile import NamedTemporaryFile
import pytest as pt

from rdds.phen2gen.dataset.schema import _SCHEMA

def test_schema():
    graph_schema = text_format.Merge(_SCHEMA, schema_pb2.GraphSchema())
    graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)


def test_generate_dummy_samples():
    graph_schema = text_format.Merge(_SCHEMA, schema_pb2.GraphSchema())
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    temporary_record_file = NamedTemporaryFile(dir='/tmp', suffix='_test_records.tfrecord')
    output_file_path = temporary_record_file.name
    with tf.io.TFRecordWriter(output_file_path) as writer:
        for _ in range(10):
            graph = tfgnn.random_graph_tensor(graph_spec)
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())


@pt.mark.skip()
def test_generate_single_graph():
    # Test write out some dummy data based on schema
    graph_schema = text_format.Merge(_SCHEMA, schema_pb2.GraphSchema())
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)
    graph = tfgnn.random_graph_tensor(graph_spec)
    print(tfgnn.write_example(graph))