import tensorflow as tf
from typing import List

from .custom_metrics import MetricSpec


class KerasCustomMetricModel(tf.keras.Model):

    def __init__(self, *args, metric_specs: List[MetricSpec], **kwargs):
        # tf.keras.metric.Metric subclasses must be instantiated before tf.keras.Model so that they're properly
        # managed in the training-eval loop. Otherwise, strange things happens, like they're not reset on epoch end...
        inputs: List[tf.Tensor] = kwargs['inputs']
        custom_metrics: List = []
        for metric_spec in metric_specs:
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

