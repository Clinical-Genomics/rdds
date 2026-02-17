import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf.text_format import Merge

_GRAPH_SCHEMA = """
node_sets {
    key: "A"
    value {
        description: "Somedesc"
        features {
            key: "A0"
            value: {
                description: "Somedesc"
                dtype: DT_STRING
            }
        }
    }
}
node_sets {
    key: "B"
    value {
        description: "Somedesc"
        features {
            key: "B0"
            value: {
                dtype: DT_FLOAT
            }
        }
    }
}
node_sets {
    key: "C"
}

edge_sets {
    key: "A>B"
    value {
        description: "Somedesc"
        source: "A"
        target: "B"

        features {
            key: "relatedness"
            value: {
                dtype: DT_FLOAT
            }
        }
    }
}

"""
_GRAPH_SCHEMA = Merge(_GRAPH_SCHEMA, schema_pb2.GraphSchema())
_GRAPH_SPEC = tfgnn.create_graph_spec_from_schema_pb(_GRAPH_SCHEMA, indices_dtype=tf.int64)
_SINGLE_GRAPH = (1, 2)


def test():
    """
    Test for generating #id feature in NodeSets with populated features.
    """
    from rdds.lib.tfgnn import random_graph_tensor_with_id

    graph = random_graph_tensor_with_id(spec=_GRAPH_SPEC,
                                        num_components_range=_SINGLE_GRAPH,
                                        validate=True)
    assert isinstance(graph, tfgnn.GraphTensor)
    assert '#id' in graph.node_sets['A'].features.keys()
    assert graph.node_sets['A'].features['#id'].dtype == tf.int64
    assert '#id' in graph.node_sets['B'].features.keys()
    assert graph.node_sets['B'].features['#id'].dtype == tf.int64
    assert '#id' not in graph.node_sets['C'].features.keys()
