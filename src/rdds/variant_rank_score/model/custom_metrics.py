import abc
from typing import Type, List, Tuple, Dict, Union
import tensorflow as tf
from dataclasses import dataclass, field
# https://keras.io/api/metrics/#as-subclasses-of-metric-stateful

@dataclass
class MetricSpec:
    InputTensorName: Union[str, dict]
    MetricClass: Type[tf.keras.metrics.Metric]
    Args: Tuple = tuple()
    Kwargs: Dict = field(default_factory=lambda: dict())


class ConfusionMatrixTracker(tf.keras.metrics.Metric, abc.ABC):

    def __init__(self, *args, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self._tps: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'{self.name}TPs',
            dtype=tf.int32
        )
        self._tns: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'{self.name}TNs',
            dtype=tf.int32
        )
        self._fns: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'{self.name}FNs',
            dtype=tf.int32
        )
        self._fps: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'{self.name}FPs',
            dtype=tf.int32
        )

    def update_state(self, y, y_pred, sample_weight=None):
        del sample_weight
        # Check inputs
        tf.debugging.assert_equal(tf.size(y), tf.size(y_pred))
        # Ground truth
        ground_truth_positives = tf.cast(y, tf.bool)
        ground_truth_negatives = tf.math.logical_not(ground_truth_positives)
        # Predictions
        prediction_positives = tf.math.greater_equal(y_pred, tf.constant(self.threshold, dtype=tf.float32))
        prediction_negatives = tf.math.logical_not(prediction_positives)
        # Confusion matrix
        tps = tf.size(tf.where(tf.math.logical_and(ground_truth_positives, prediction_positives))[:, 0])
        self._tps.assign_add(tps)
        tns = tf.size(tf.where(tf.math.logical_and(ground_truth_negatives, prediction_negatives))[:, 0])
        self._tns.assign_add(tns)
        fns = tf.size(tf.where(tf.math.logical_and(ground_truth_positives, prediction_negatives))[:, 0])
        self._fns.assign_add(fns)
        fps = tf.size(tf.where(tf.math.logical_and(ground_truth_negatives, prediction_positives))[:, 0])
        self._fps.assign_add(fps)

    @staticmethod
    def _as_float(v: tf.Variable) -> tf.Tensor:
        return tf.cast(v, tf.float32)

    @property
    def tps(self):
        return self._as_float(self._tps)

    @property
    def tns(self):
        return self._as_float(self._tns)

    @property
    def fps(self):
        return self._as_float(self._fps)

    @property
    def fns(self):
        return self._as_float(self._fns)

    @abc.abstractmethod
    def result(self) -> tf.Tensor:
        """
        Compute metric score in subclass.
        """
        pass

    def reset_state(self):
        super().reset_state()
        self._tps.assign(0)
        self._tns.assign(0)
        self._fns.assign(0)
        self._fps.assign(0)


class MccScore(ConfusionMatrixTracker):
    def __init__(self, *args, name='MCC', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self) -> tf.Tensor:
        numerator = (self.tps * self.tns) - (self.fps * self.fns)
        denominator = ((self.tps + self.fps) * (self.tps + self.fns) * (self.tns + self.fps) * (self.tns + self.fns)) ** 0.5
        return tf.math.divide_no_nan(numerator, denominator)


class F1Score(ConfusionMatrixTracker):
    def __init__(self, *args, name='F1', **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self) -> tf.Tensor:
        numerator = 2 * self.tps
        denominator = (2 * self.tps) + self.fps + self.fns
        return tf.math.divide_no_nan(numerator, denominator)


class RegexpF1(F1Score):

    """
    A metric that collects TPs and FNs matching a particular tensor and regexp to compute F1 score.
    Yields no indication of FPs (ideally) since all collected samples are ground truth positive.
    """

    # https://regex101.com/

    def __init__(self, *args,
                 tensor_idx: int,
                 pattern: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._tensor_idx = tensor_idx
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_samples_{self.name}'
        )
        self._pattern = pattern

    def update_state(self, x, y, y_pred, sample_weight=None):
        data_tensor = x[self._tensor_idx]
        labels = y[0]  # -> [batch_size, ]
        regexp_matches = tf.strings.regex_full_match(input=data_tensor,
                                                     pattern=self._pattern,
                                                     name=f'{self.name}RegexpMatch')
        matching_idx = tf.where(regexp_matches)[:, 0]
        matching_labels = tf.gather(labels, matching_idx)
        matching_predictions = tf.gather(y_pred, matching_idx)
        n_samples = tf.cast(tf.size(matching_labels), tf.float32)
        super().update_state(y=matching_labels, y_pred=matching_predictions)
        with tf.control_dependencies([n_samples]):
            self.total_samples.assign_add(n_samples)

    def result(self):
        return {
            f'{self.name}': super().result(),
            f'{self.name}Count': self.total_samples
        }

    def reset_state(self):
        super().reset_state()
        self.total_samples.assign(0)


class FrequencyFilteredF1(F1Score):
    # TODO: Merge multiple tensors and compute average frq?
    def __init__(self,
                 *args,
                 tensor_idx: int,
                 frequency_threshold: float = 1.0 / 2000.0,
                 frequency_filter_method: str = 'less_equal',
                 ignore_zero_padded_values: bool = True,
                 **kwargs):
        """
        :param args: subclass
        :param tensor_idx: Index of tensor containing variant population frequencies
        :param frequency_threshold: Cutoff frequency to incorporate a variant in subset for metric computation
        :param frequency_filter_method: Comparison method to incorporate variant in subset for metric computation
        :param ignore_zero_padded_values: Ignore variant if 0.0 is detected in frq tensor (zero padded, lacking annotation)
        :param kwargs: subclass
        """
        super().__init__(*args, **kwargs)
        self._tensor_idx = tensor_idx
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_samples_{self.name}'
        )
        self._ignore_zero_padded_values = ignore_zero_padded_values
        self.frequency_threshold = tf.constant(frequency_threshold, dtype=tf.float32)
        self.frequency_filter_method = frequency_filter_method

    def update_state(self, x, y, y_pred, sample_weight):
        frq_tensor = x[self._tensor_idx]
        labels = y[0]  # -> [batch_size, ]
        # Find indexes of variants with a particular occurrence frequency in population
        if self.frequency_filter_method == 'less_equal':
            frq_cond = tf.math.less_equal(frq_tensor, self.frequency_threshold)  # to boolean vector
        elif self.frequency_filter_method == 'greater':
            frq_cond = tf.math.greater(frq_tensor, self.frequency_threshold)
        else:
            raise ValueError(f'Unsupported frequency filter method: {self.frequency_filter_method}')
        if self._ignore_zero_padded_values:
            is_not_zero_padded_cond = tf.math.greater(frq_tensor, tf.constant(0, dtype=tf.float32))
            frq_cond = tf.math.logical_and(frq_cond, is_not_zero_padded_cond)
        idx = tf.where(frq_cond)[:, 0]  # to index vector (flattened)
        # Select labels and predictions and update internal counters
        filtered_labels = tf.gather(labels, idx)
        filtered_predictions = tf.gather(y_pred, idx)
        super().update_state(y=filtered_labels, y_pred=filtered_predictions)
        n_samples = tf.cast(tf.size(filtered_labels), tf.float32)
        self.total_samples.assign_add(n_samples)

    def result(self) -> dict:
        return {
            f'{self.name}': super().result(),
            f'{self.name}Count': self.total_samples
        }

    def reset_state(self):
        super().reset_state()
        self.total_samples.assign(0)


class RareVariantWithoutClinvarSupportF1(F1Score):
    def __init__(self,
                 *args,
                 tensor_idx_frq: int,
                 ignore_zero_padded_values: bool = True,
                 tensor_idx_clinvar_clnsig: int,
                 **kwargs):
        """
        :param args: subclass
        :param tensor_idx_frq: Index of tensor containing variant population frequencies
        :param tensor_idx_clinvar_clnsig: Index of tensor containing CLINVAR CLNSIG annotations
        :param ignore_zero_padded_values: Ignore variant if 0.0 is detected in frq tensor (zero padded, lacking annotation)
        :param kwargs: subclass
        """
        super().__init__(*args, **kwargs)
        self._tensor_idx_frq = tensor_idx_frq
        self._tensor_idx_clinvar_clnsig = tensor_idx_clinvar_clnsig
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_samples_{self.name}'
        )
        self._ignore_zero_padded_values = ignore_zero_padded_values

    def update_state(self, x, y, y_pred, sample_weight):
        frq_tensor = x[self._tensor_idx_frq]
        clinvar_clnsig_tensor = x[self._tensor_idx_clinvar_clnsig]
        labels = y[0]  # -> [batch_size, ]
        # Find indexes of rare vs common data points
        is_rare_variant_cond = tf.math.less_equal(frq_tensor, tf.constant(1.0/2000.0, dtype=tf.float32))  # to boolean vector
        if self._ignore_zero_padded_values:
            is_not_zero_padded_cond = tf.math.greater(frq_tensor, tf.constant(0, dtype=tf.float32))
            is_rare_variant_cond = tf.math.logical_and(is_rare_variant_cond, is_not_zero_padded_cond)
        no_clinvar_support_cond = tf.strings.regex_full_match(input=clinvar_clnsig_tensor,
                                                              pattern='^\s*$',  # empty string or whitespaces only
                                                              name=f'{self.name}RegexpMatch')
        rare_and_no_clinvar_support_cond = tf.math.logical_and(is_rare_variant_cond, no_clinvar_support_cond)
        idx = tf.where(rare_and_no_clinvar_support_cond)[:, 0]  # to index vector (flattened)
        # Subset data
        filtered_labels = tf.gather(labels, idx)
        filtered_predictions = tf.gather(y_pred, idx)
        # Update internal counters
        super().update_state(y=filtered_labels, y_pred=filtered_predictions)
        n_samples = tf.cast(tf.size(filtered_labels), tf.float32)
        self.total_samples.assign_add(n_samples)

    def result(self) -> dict:
        return {
            f'{self.name}': super().result(),
            f'{self.name}Count': self.total_samples,
        }

    def reset_state(self):
        super().reset_state()
        self.total_samples.assign(0)


def custom_metrics(extended_vep_metrics: bool = False) -> List[MetricSpec]:
    """
    Generate custom metric spec.
    Adds additional load to training procedure, so this is mainly on a debug need basis only.
    :param extended_vep_metrics: Add metrics to stratify performance based on VEP annotation
    """
    custom_metrics_list: List[MetricSpec] = []
    vep_consequence_terms = \
    ['missense_variant', 'downstream_gene_variant', 'upstream_gene_variant',
     'intron_variant', 'non_coding_transcript_exon_variant',
     'splice_donor_variant', 'splice_donor_region_variant',
     'splice_region_variant', '5_prime_UTR_variant',
     'splice_polypyrimidine_tract_variant', '3_prime_UTR_variant',
     'synonymous_variant', 'frameshift_variant', 'stop_gained',
     'splice_acceptor_variant', 'splice_donor_5th_base_variant', 'stop_lost',
     'protein_altering_variant', 'inframe_insertion', 'inframe_deletion',
     'transcript_ablation', 'start_lost', 'stop_retained_variant',
     'coding_sequence_variant', 'mature_miRNA_variant',
     'incomplete_terminal_codon_variant']
    if extended_vep_metrics:
        for term in vep_consequence_terms:
            custom_metrics_list.append(
                MetricSpec(InputTensorName='most_severe_consequence',
                MetricClass=RegexpF1,
                Kwargs={'pattern': f'.*({term}).*', 'name': f'{term}F1'})
            )
    custom_metrics_list.append(
        MetricSpec(InputTensorName='GNOMADAF_popmax',
        MetricClass=FrequencyFilteredF1,
        Kwargs={'name': 'RareVariantF1Gnomad'}))
    custom_metrics_list.append(
        MetricSpec(InputTensorName='GNOMADAF_popmax',
        MetricClass=FrequencyFilteredF1,
        Kwargs={'name': 'CommonVariantF1Gnomad', 'frequency_filter_method': 'greater'}))
    custom_metrics_list.append(
        MetricSpec(InputTensorName='Frq',
        MetricClass=FrequencyFilteredF1,
        Kwargs={'name': 'RareVariantF1Frq'}))
    custom_metrics_list.append(
        MetricSpec(InputTensorName='Frq',
        MetricClass=FrequencyFilteredF1,
        Kwargs={'name': 'CommonVariantF1Frq', 'frequency_filter_method': 'greater'}))
    custom_metrics_list.append(
        MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
        MetricClass=RegexpF1,
        Kwargs={'pattern': '.*(pathogenic|Pathogenic).*',
        'name': 'ClinvarPathogenicF1'}))
    custom_metrics_list.append(
        MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
        MetricClass=RegexpF1,
        Kwargs={'pattern': '.*(benign|Benign).*',
        'name': 'ClinvarBenignF1'}))
    custom_metrics_list.append(
        MetricSpec(
            InputTensorName={'tensor_idx_frq': 'GNOMADAF_popmax',
                             'tensor_idx_clinvar_clnsig': 'CSQ_CLINVAR_CLNSIG'},
            MetricClass=RareVariantWithoutClinvarSupportF1,
            Kwargs={'name': 'RareVariantWithoutClinvarSupportF1'})
    )
    return custom_metrics_list