import tensorflow_gnn as tfgnn

def generate_graph_from_example(record_bytes):
        graph = tfgnn.parse_single_example(
            graph_tensor_spec, record_bytes, validate=True)

        # extract label from context and remove from input graph
        context_features = graph.context.get_features_dict()
        label = context_features.pop('label')
        new_graph = graph.replace_features(context=context_features)

        return new_graph, label