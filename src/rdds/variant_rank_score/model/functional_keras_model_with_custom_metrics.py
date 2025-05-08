import tensorflow as tf
from typing import List
import pickle
from copy import deepcopy
from keras.engine.functional import reconstruct_from_config, connect_ancillary_layers, get_network_config

from .custom_metrics import MetricSpec


class FunctionalKerasModelWithCustomMetrics(tf.keras.Model):

    def __init__(self, *args, metric_specs: List[MetricSpec], **kwargs):
        # tf.keras.metric.Metric subclasses must be instantiated before tf.keras.Model so that they're properly
        # managed in the training-eval loop. Otherwise, strange things happens, like they're not reset on epoch end...
        self._metric_specs = metric_specs
        inputs = kwargs.get('inputs')
        custom_metrics: List = []
        for metric_spec in self._metric_specs:
            if isinstance(metric_spec.InputTensorName, str):
                metric_spec.Kwargs.update({'tensor_idx':
                                           self._get_tensor_input_index_from_name(name=metric_spec.InputTensorName,
                                                                                  inputs=inputs)})
            elif isinstance(metric_spec.InputTensorName, dict):
                tensor_kwargs = {}
                for kwarg, tensor_name in metric_spec.InputTensorName.items():
                    tensor_kwargs.update({kwarg: self._get_tensor_input_index_from_name(name=tensor_name,
                                                                                        inputs=inputs)})
                metric_spec.Kwargs.update(tensor_kwargs)
            else:
                raise ValueError(f'Unsupported InputTensorName Type {type(metric_spec.InputTensorName)}')
            custom_metrics.append(metric_spec.MetricClass(
                *metric_spec.Args,
                **metric_spec.Kwargs
            ))
        super().__init__(*args, **kwargs)
        self._custom_metrics = custom_metrics

    @staticmethod
    def _get_tensor_input_index_from_name(name: str,
                                          inputs: List[tf.Tensor]) -> int:
        """
        Helper method to return idx of input tensor.
        :param name: A tensor name
        :return: Index of tensor in input array
        """
        for idx, input_tensor in enumerate(inputs):
            if name == input_tensor.name:
                return idx
        raise ValueError(f'No matching input tensor with name {name} in {inputs}')

    def compute_metrics(self, x, y, y_pred, sample_weight) -> dict:
        """
        Compute custom metrics and log to Tensorboard.
        Overrides tf.keras.Model.compute_metrics()
        :param x: training data
        :param y: training labels
        :param y_pred: model predictions
        :param sample_weight: sample weight
        :return: Updated metrics dict
        """
        # Compute "standard" metrics supplied in the metrics argument to self.compile()
        metric_results: dict = super().compute_metrics(x, y, y_pred, sample_weight)
        # Compute custom metrics
        for custom_metric in self._custom_metrics:
            custom_metric.update_state(x, y, y_pred, sample_weight)
            metric_results.update(custom_metric.result())
        return metric_results

    def get_config(self) -> dict:
        """
        Serialisation/deserialization is now a special case since this class subclasses keras.Model
        but still making use of the keras Functional API for graph definition.

        When creating the config, make sure to use the Functional API configuration
        (contrasting keras.Model subclass API which contains no information).
        """
        # Subclass (keras.engine.functional.Functional class config)
        functional_configs = get_network_config(self)
        # Current class config
        metric_spec: str = pickle.dumps(self._metric_specs).hex()
        subclass_configs = super().get_config()
        subclass_configs.update({'metric_specs': metric_spec})
        final_config = {
            'functional_configs': functional_configs,
            'subclass_configs': subclass_configs
        }
        return final_config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        """
        Restoring this class is a special case, since we cannot use the
        keras.Model API (model was created using the Functional API)
        but we still have to pass the current class args, kwargs somehow.

        Do this by restoring the model args, kwargs using Functional API
        and then re-create the class from a separate set of args, kwargs.
        """

        # Re-instantiate args, kwargs for Functional model
        functional_configs = deepcopy(config['functional_configs'])
        inputs, outputs, layers = reconstruct_from_config(
            functional_configs, custom_objects
        )

        # Re-instantiate args, kwargs for current class and keras.Model subclass.
        # No other args, kwargs apart from Functional API keywords are allowed
        # (otherwise keras.Model __init__ triggers the wrong init as Model subclass).
        subclass_configs = deepcopy(config['subclass_configs'])
        metric_specs: bytes = bytes.fromhex(subclass_configs['metric_specs'])
        metric_specs = pickle.loads(metric_specs)  # type:ListWrapper
        metric_specs: List[MetricSpec] = list(metric_specs)
        subclass_configs.update({'metric_specs': metric_specs})

        # Now, once all args, kwargs have been restored, recreate this class
        # using keras.Model Functional API logic.
        model = FunctionalKerasModelWithCustomMetrics(inputs=inputs,
                                                      outputs=outputs,
                                                      name=functional_configs.get("name"),
                                                      metric_specs=metric_specs
        )

        # ... and finally connect dangling layers to the model object
        connect_ancillary_layers(model, layers)

        return model
