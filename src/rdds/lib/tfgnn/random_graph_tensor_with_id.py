from tensorflow import int64, range as tfrange
import tensorflow_gnn as tfgnn


def random_graph_tensor_with_id(*args, **kwargs) -> tfgnn.GraphTensor:
    """
    Add #id feature to NodeSets that have set Features, so that tgfnn samplers can access node -> feature positions.
    """
    graph_tensor = tfgnn.random_graph_tensor(*args, **kwargs)
    for node_set_name in graph_tensor.node_sets:
        node_set_name: str
        node_set: tfgnn.NodeSet = graph_tensor.node_sets[node_set_name]
        if len(node_set.features) > 0:
            ids = tfrange(0, node_set.total_size, dtype=int64)
            new_features = {
                "#id": ids
            }
            new_features.update(node_set.features)
            # Update nodeset_name/features
            graph_tensor = graph_tensor.replace_features(node_sets={node_set_name: new_features})
    return graph_tensor
