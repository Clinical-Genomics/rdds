from typing import Type, List, Tuple, Dict, Union
import tensorflow as tf
from dataclasses import dataclass, field
# https://keras.io/api/metrics/#as-subclasses-of-metric-stateful
from rdds.lib.tf import mcc, f1


@dataclass
class MetricSpec:
    InputTensorName: Union[str, dict]
    MetricClass: Type[tf.keras.metrics.Metric]
    Args: Tuple = tuple()
    Kwargs: Dict = field(default_factory=lambda: dict())


class MccScore(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name="MCC", dtype=None, threshold=0.5):
        super().__init__(
            mcc, name, dtype=dtype, threshold=threshold
        )


class F1Score(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name="F1", dtype=None, threshold=0.5):
        super().__init__(
            f1, name, dtype=dtype, threshold=threshold
        )


class RegexpF1(tf.keras.metrics.Metric):

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

        self.binary_f1: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'f1_{self.name}'
        )
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_{self.name}'
        )
        self._pattern = pattern

    def update_state(self, x, y, y_pred, sample_weight):
        data_tensor = x[self._tensor_idx]
        labels = y[0]  # -> [batch_size, ]
        regexp_matches = tf.strings.regex_full_match(input=data_tensor,
                                                     pattern=self._pattern,
                                                     name=f'{self.name}RegexpMatch')
        matching_idx = tf.where(regexp_matches)[:, 0]
        matching_labels = tf.gather(labels, matching_idx)
        matching_predictions = tf.gather(y_pred, matching_idx)
        score = f1(matching_labels, matching_predictions)
        self.binary_f1.assign_add(score)
        self.total_samples.assign_add(tf.cast(tf.size(matching_labels), tf.float32))

    def result(self):
        return {
            f'{self.name}': tf.math.divide_no_nan(self.binary_f1, self.total_samples),
            f'{self.name}Count': self.total_samples
        }

    def reset_state(self):
        self.binary_f1.assign(0)
        self.total_samples.assign(0)


class RareVariantF1(tf.keras.metrics.Metric):
    # TODO: Merge multiple tensors and compute average frq?
    def __init__(self,
                 *args,
                 tensor_idx: int,
                 ignore_zero_padded_values: bool = True,
                 **kwargs):
        """
        :param args: subclass
        :param tensor_idx: Index of tensor containing variant population frequencies
        :param ignore_zero_padded_values: Ignore variant if 0.0 is detected in frq tensor (zero padded, lacking annotation)
        :param kwargs: subclass
        """
        super().__init__(*args, **kwargs)
        self._tensor_idx = tensor_idx

        self.binary_f1_rare: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_f1_rare_{self.name}'
        )
        self.binary_f1_common: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_f1_common_{self.name}'
        )
        self.total_rare_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_rare_samples_{self.name}'
        )
        self.total_common_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_common_samples_{self.name}'
        )
        self._ignore_zero_padded_values = ignore_zero_padded_values

    def update_state(self, x, y, y_pred, sample_weight):
        frq_tensor = x[self._tensor_idx]
        labels = y[0]  # -> [batch_size, ]
        # Find indexes of rare vs common data points
        is_rare_variant_cond = tf.math.less_equal(frq_tensor, tf.constant(1.0/2000.0, dtype=tf.float32))  # to boolean vector
        if self._ignore_zero_padded_values:
            is_not_zero_padded_cond = tf.math.greater(frq_tensor, tf.constant(0, dtype=tf.float32))
            is_rare_variant_cond = tf.math.logical_and(is_rare_variant_cond, is_not_zero_padded_cond)
        is_common_variant_cond = tf.math.logical_not(is_rare_variant_cond)
        rare_idx = tf.where(is_rare_variant_cond)[:, 0]  # to index vector (flattened)
        common_idx = tf.where(is_common_variant_cond)[:, 0]
        # Rare data
        rare_labels = tf.gather(labels, rare_idx)
        y_pred_rare = tf.gather(y_pred, rare_idx)
        # Common data
        common_labels = tf.gather(labels, common_idx)
        y_pred_common = tf.gather(y_pred, common_idx)
        # Compute f1
        rare_f1 = f1(rare_labels, y_pred_rare)
        self.binary_f1_rare.assign_add(rare_f1)
        self.total_rare_samples.assign_add(tf.cast(tf.size(rare_labels), tf.float32))
        common_f1 = f1(common_labels, y_pred_common)
        self.binary_f1_common.assign_add(common_f1)
        self.total_common_samples.assign_add(tf.cast(tf.size(common_labels), tf.float32))

    def result(self) -> dict:
        mean_pathogenic = tf.math.divide_no_nan(self.binary_f1_rare, self.total_rare_samples)
        mean_benign = tf.math.divide_no_nan(self.binary_f1_common, self.total_common_samples)
        return {
            f'{self.name}Rare': mean_pathogenic,
            f'{self.name}Common': mean_benign,
            f'{self.name}RareCount': self.total_rare_samples,
            f'{self.name}CommonCount': self.total_common_samples,
        }

    def reset_state(self):
        self.binary_f1_rare.assign(0)
        self.binary_f1_common.assign(0)
        self.total_rare_samples.assign(0)
        self.total_common_samples.assign(0)


class RareVariantWithoutClinvarSupportF1(tf.keras.metrics.Metric):
    def __init__(self,
                 *args,
                 tensor_idx_frq: int,
                 tensor_idx_clinvar_clnsig: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._tensor_idx_frq = tensor_idx_frq
        self._tensor_idx_clinvar_clnsig = tensor_idx_clinvar_clnsig

        self.binary_f1: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_f1{self.name}'
        )
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_samples_{self.name}'
        )

    def update_state(self, x, y, y_pred, sample_weight):
        frq_tensor = x[self._tensor_idx_frq]
        clinvar_clnsig_tensor = x[self._tensor_idx_clinvar_clnsig]
        labels = y[0]  # -> [batch_size, ]
        # Find indexes of rare vs common data points
        is_rare_variant_cond = tf.math.less_equal(frq_tensor, tf.constant(1.0/2000.0, dtype=tf.float32))  # to boolean vector
        no_clinvar_support_cond = tf.strings.regex_full_match(input=clinvar_clnsig_tensor,
                                                              pattern='^\s*$',  # empty string or whitespaces only
                                                              name=f'{self.name}RegexpMatch')
        rare_and_no_clinvar_support_cond = tf.math.logical_and(is_rare_variant_cond, no_clinvar_support_cond)
        idx = tf.where(rare_and_no_clinvar_support_cond)[:, 0]  # to index vector (flattened)
        # Subset data
        labels_subset = tf.gather(labels, idx)
        y_pred_subset = tf.gather(y_pred, idx)
        # Compute f1
        score = f1(labels_subset, y_pred_subset)
        self.binary_f1.assign_add(score)
        self.total_samples.assign_add(tf.cast(tf.size(labels_subset), tf.float32))

    def result(self) -> dict:
        mean = tf.math.divide_no_nan(self.binary_f1, self.total_samples)
        return {
            f'{self.name}': mean,
            f'{self.name}Count': self.total_samples,
        }

    def reset_state(self):
        self.binary_f1.assign(0)
        self.total_samples.assign(0)


CUSTOM_METRICS: List[MetricSpec] = []
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
for term in vep_consequence_terms:
    CUSTOM_METRICS.append(
        MetricSpec(InputTensorName='most_severe_consequence',
        MetricClass=RegexpF1,
        Kwargs={'pattern': f'.*({term}).*', 'name': f'{term}F1'})
    )
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='GNOMADAF_popmax',
    MetricClass=RareVariantF1,
    Kwargs={'name': 'RareVariantF1Gnomad'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='Frq',
    MetricClass=RareVariantF1,
    Kwargs={'name': 'RareVariantF1Frq'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
    MetricClass=RegexpF1,
    Kwargs={'pattern': '.*(pathogenic|Pathogenic).*',
    'name': 'ClinvarPathogenicF1'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
    MetricClass=RegexpF1,
    Kwargs={'pattern': '.*(benign|Benign).*',
    'name': 'ClinvarBenignF1'}))
CUSTOM_METRICS.append(
    MetricSpec(
        InputTensorName={'tensor_idx_frq': 'GNOMADAF_popmax',
                         'tensor_idx_clinvar_clnsig': 'CSQ_CLINVAR_CLNSIG'},
        MetricClass=RareVariantWithoutClinvarSupportF1,
        Kwargs={'name': 'RareVariantWithoutClinvarSupportF1'})
)