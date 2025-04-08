from typing import Type, List, Tuple, Dict, Union
import tensorflow as tf
from dataclasses import dataclass, field
# https://keras.io/api/metrics/#as-subclasses-of-metric-stateful
from rdds.lib.tf import mcc


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


class RegexpMCC(tf.keras.metrics.Metric):

    # https://regex101.com/

    def __init__(self, *args,
                 tensor_idx: int,
                 pattern: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._tensor_idx = tensor_idx

        self.binary_accuracy: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'mcc_{kwargs["name"]}'
        )
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_{kwargs["name"]}'
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
        score = mcc(matching_labels, matching_predictions)
        self.binary_accuracy.assign_add(score)
        self.total_samples.assign_add(tf.cast(tf.size(matching_labels), tf.float32))

    def result(self):
        return {
            f'{self.name}': tf.math.divide_no_nan(self.binary_accuracy, self.total_samples),
            f'Count{self.name}': self.total_samples
        }

    def reset_state(self):
        self.binary_accuracy.assign(0)
        self.total_samples.assign(0)


class RareVariantMCC(tf.keras.metrics.Metric):
    def __init__(self, *args, tensor_idx: int, name='RareVariantBinaryMCC', **kwargs):
        kwargs.update({'name': name})
        super().__init__(*args, **kwargs)
        self._tensor_idx = tensor_idx

        self.binary_accuracy_pathogenic: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_accuracy_pathogenic_{name}'
        )
        self.binary_accuracy_benign: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_accuracy_benign_{name}'
        )
        self.total_pathogenic_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_pathogenic_samples_{name}'
        )
        self.total_benign_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_benign_samples_{name}'
        )

    def update_state(self, x, y, y_pred, sample_weight):
        frq_tensor = x[self._tensor_idx]
        labels = y[0]  # -> [batch_size, ]
        # Find indexes of rare vs common data points
        is_rare_variant_cond = tf.math.less_equal(frq_tensor, tf.constant(1.0/2000.0, dtype=tf.float32))  # to boolean vector
        is_common_variant_cond = tf.math.logical_not(is_rare_variant_cond)
        rare_idx = tf.where(is_rare_variant_cond)[:, 0]  # to index vector (flattened)
        common_idx = tf.where(is_common_variant_cond)[:, 0]
        # Rare data
        rare_labels = tf.gather(labels, rare_idx)
        y_pred_rare = tf.gather(y_pred, rare_idx)
        # Common data
        common_labels = tf.gather(labels, common_idx)
        y_pred_common = tf.gather(y_pred, common_idx)
        # Compute mcc
        rare_mcc = mcc(rare_labels, y_pred_rare)
        self.binary_accuracy_pathogenic.assign_add(rare_mcc)
        self.total_pathogenic_samples.assign_add(tf.cast(tf.size(rare_labels), tf.float32))
        common_mcc = mcc(common_labels, y_pred_common)
        self.binary_accuracy_benign.assign_add(common_mcc)
        self.total_benign_samples.assign_add(tf.cast(tf.size(common_labels), tf.float32))

    def result(self) -> dict:
        mean_pathogenic = tf.math.divide_no_nan(self.binary_accuracy_pathogenic, self.total_pathogenic_samples)
        mean_benign = tf.math.divide_no_nan(self.binary_accuracy_benign, self.total_benign_samples)
        return {
            f'Pathogenic{self.name}': mean_pathogenic,
            f'Benign{self.name}': mean_benign,
            f'CountPathogenic{self.name}': self.total_benign_samples,
            f'CountBenign{self.name}': self.total_pathogenic_samples,
        }

    def reset_state(self):
        self.binary_accuracy_pathogenic.assign(0)
        self.binary_accuracy_benign.assign(0)
        self.total_pathogenic_samples.assign(0)
        self.total_benign_samples.assign(0)


class RareVariantWithoutClinvarSupportMCC(tf.keras.metrics.Metric):
    def __init__(self,
                 *args,
                 tensor_idx_frq: int,
                 tensor_idx_clinvar_clnsig: int,
                 name='RareVariantWithoutClinvarSupportMCC', **kwargs):
        kwargs.update({'name': name})
        super().__init__(*args, **kwargs)
        self._tensor_idx_frq = tensor_idx_frq
        self._tensor_idx_clinvar_clnsig = tensor_idx_clinvar_clnsig

        self.binary_accuracy: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'binary_accuracy{name}'
        )
        self.total_samples: tf.Variable = self.add_variable(
            shape=(),
            initializer='zeros',
            name=f'total_samples_{name}'
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
        # Compute mcc
        score = mcc(labels_subset, y_pred_subset)
        self.binary_accuracy.assign_add(score)
        self.total_samples.assign_add(tf.cast(tf.size(labels_subset), tf.float32))

    def result(self) -> dict:
        mean = tf.math.divide_no_nan(self.binary_accuracy, self.total_samples)
        return {
            f'{self.name}': mean,
            f'Count{self.name}': self.total_samples,
        }

    def reset_state(self):
        self.binary_accuracy.assign(0)
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
        MetricClass=RegexpMCC,
        Kwargs={'pattern': f'.*({term}).*', 'name': f'{term}Accuracy'})
    )
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='GNOMADAF_popmax',
    MetricClass=RareVariantMCC,
    Kwargs={'name': 'RareVariantMCCGnomad'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='Frq',
    MetricClass=RareVariantMCC,
    Kwargs={'name': 'RareVariantMCCFrq'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
    MetricClass=RegexpMCC,
    Kwargs={'pattern': '.*(pathogenic|Pathogenic).*',
    'name': 'ClinvarPathogenicMCC'}))
CUSTOM_METRICS.append(
    MetricSpec(InputTensorName='CSQ_CLINVAR_CLNSIG',
    MetricClass=RegexpMCC,
    Kwargs={'pattern': '.*(benign|Benign).*',
    'name': 'ClinvarBenignMCC'}))
CUSTOM_METRICS.append(
    MetricSpec(
        InputTensorName={'tensor_idx_frq': 'GNOMADAF_popmax',
                         'tensor_idx_clinvar_clnsig': 'CSQ_CLINVAR_CLNSIG'},
        MetricClass=RareVariantWithoutClinvarSupportMCC)
)