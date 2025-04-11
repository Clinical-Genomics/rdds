# Make sure to set all seeds
import pandas as pd

from rdds.lib.determinism import enable_determinism; enable_determinism()

# Add NaN, +-Inf checks
from rdds.lib.tf import enable_check_numerics; enable_check_numerics()

from typing import List, Dict, Union, Type, Tuple, Optional, Set, Any
import numpy as np
import tensorflow as tf
import os
import datetime
from json import dumps
import logging
from dataclasses import dataclass
from h5py import File as Hdf5File, string_dtype
import gc

from .. import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.hdf5 import Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator
from rdds.lib.tf import TextPreprocessingLayer
from rdds.lib.tf import EmbeddingsReductionLayer
from rdds.lib.tf import DnaSequenceTrimmer
from rdds.lib.tf import InstanceNormalisationLayer
from rdds.lib.tf.augmented_dropout_dataset import TextAugmentDropoutDataset
from ..dataset.class_labels import LABEL_PATHOGENIC_VARIANT, LABEL_BENIGN_VARIANT
from rdds.lib.tf import rejection_resample
from rdds.lib.tf import print_tensor_op
from rdds.lib.hpt import HyperParameters
from rdds.lib.vcf import ParsableVariant
from .model_explainer import ModelExplainer
from .default_model import DEFAULT_MODEL_SPEC
from .keras_custom_metric_model import KerasCustomMetricModel, MetricSpec
from .custom_metrics import CUSTOM_METRICS, MccScore, F1Score



@dataclass
class InitializedDatasets:
    dataset_train: tf.data.Dataset
    dataset_test: tf.data.Dataset
    train_data_length: int
    test_data_length: int
    batch_size: int
    model_bias_estimate: float  # Estimate bias for model output layer, computed from training data class ratio
    dataset_train_numerical: tf.data.Dataset = None
    dataset_train_vocabulary:  tf.data.Dataset = None


"""
NOTE!
Always make sure to keep a reference to all layers created in the model, i.e.
layer = tf.keras.layers.MyLayer()
y = layer(x)
If this is not the case, the layer might be removed in the graph!
"""

# See comment in the text_vectorization_layer.py about keras and tensorboard.
_LOGGER = get_logger('vrs-model')
_LOGGER.setLevel(logging.INFO)

FEATURES_TEXT = ['CSQ_PolyPhen',
                 'CSQ_SIFT',
                 'CSQ_CLINVAR_CLNREVSTAT',
                 'CSQ_CLINVAR_CLNSIG',
                 #'FILTER',
                 'most_severe_consequence',
                 #'GeneticModels_model'  # Dropped since clinvar does not contain pedigree information
                 ]

FEATURES_FLOAT = ['CSQ_MaxEntScan_alt',
                  'CSQ_MaxEntScan_diff',
                  'CSQ_MES-SWA_acceptor_alt',
                  'CSQ_MES-SWA_donor_alt',
                  'CSQ_MES-SWA_donor_diff',
                  'CSQ_SpliceAI_pred_DS_AL',
                  'CSQ_SpliceAI_pred_DS_DG',
                  'CSQ_SpliceAI_pred_DS_DL',
                  'CSQ_REVEL_score',
                  'CSQ_LoFtool',
                  'CSQ_GERP++_RS',
                  'CSQ_phastCons100way_vertebrate',
                  'CSQ_phyloP100way_vertebrate',
                  'CADD',
                  'ModelScore_value',
                  'SWEGENAF',
                  'GNOMADAF_popmax',
                  'SPIDEX',
                  'CSQ_SpliceAI_pred_DS_AG',
                  #'MTAF',
                  'Frq'
                  ]



class VariantRankScoreModel:

    def __init__(self,
                 features_text: List[str] = FEATURES_TEXT,
                 features_numerical: List[str] = FEATURES_FLOAT,
                 vocabulary_file_path: str = DEFAULT_MODEL_SPEC.vocabulary_file,
                 numerical_normalisation_weights_file_path: str = DEFAULT_MODEL_SPEC.numerical_normalisation_weights,
                 workdir: str = WORKDIR,
                 workdir_suffix: str = 'models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
        """

        :param features_text:
        :param features_numerical:
        :param vocabulary_file_path:
        Set to None if you wish to regenereate the vocabulary.
        Note: Training fails after vocabulary has been generated, so update this path
        and restart training.
        :param numerical_normalisation_weights_file_path:
        Set to None if you wish to regenerate the normalisation weights.
        :param workdir: Path where model build and training output is stored
        :param workdir_suffix: Subdirectory used for stratifying model training runs
        """
        self._features_text: List[str] = features_text
        self._features_numerical = features_numerical
        self._features: List[str] = self._features_text + self._features_numerical
        _LOGGER.info(f'Total amount of features: {len(self._features)}')
        self._vocabulary_file_path = vocabulary_file_path
        self._numerical_normalisation_weights_file_path = numerical_normalisation_weights_file_path
        self._keras_model: tf.keras.Model = None
        self._workdir = workdir
        self._train_log_dir: str = os.path.join(self._workdir, workdir_suffix)
        self._datasets: InitializedDatasets = None
        self._model_explainer: ModelExplainer = None

    def _build_model(self,
                     hparams: HyperParameters) -> tf.keras.models.Model:
        """
        :param hparams: Hyperparameters for the model
        """

        text_inputs = []
        for text_feature_name in self._features_text:
            text_inputs.append(
                tf.keras.Input(shape=(1, ),
                               ragged=True,
                               dtype=tf.string,
                               name=text_feature_name)
            )

        numerical_inputs = []
        for numerical_feature_name in self._features_numerical:
            numerical_inputs.append(
                tf.keras.Input(shape=(1, ),
                               dtype=tf.float32,
                               name=numerical_feature_name)
            )

        # Concatenate inputs
        input_numerical: tf.Tensor = tf.concat(numerical_inputs, axis=1, name='concat_input_numerical')
        input_text: tf.RaggedTensor = tf.concat(text_inputs, axis=1, name='concat_input_text')

        # Text preprocessing
        split_regex = '\s|\n|_|&|/|\||:|,|-|0|1|2|3|4|5|6|7|8|9'
        text_preprocessing_layer = TextPreprocessingLayer(split_regex=split_regex)
        preprocessed_dataset = \
            self._datasets.dataset_train_vocabulary.map(map_func=text_preprocessing_layer) \
            if self._datasets.dataset_train_vocabulary else None
        input_text_preprocessed = text_preprocessing_layer(input_text)  # -> [bdim, n_words, n_features]
        dna_sequence_trimmer_layer = DnaSequenceTrimmer()
        preprocessed_dataset = \
            preprocessed_dataset.map(map_func=dna_sequence_trimmer_layer) if preprocessed_dataset else None
        input_text_preprocessed = dna_sequence_trimmer_layer(input_text_preprocessed)  # tensor shape preserved

        with_feature_selection_regularisation = hparams.Boolean('feature_selection_regularisation',
                                                                default=True)
        with hparams.conditional_scope('feature_selection_regularisation',  [True]):
            if with_feature_selection_regularisation:
                # L1 regularizer to perform feature selection
                penalty = hparams.Float('feature_selection_regularisation_penalty',
                                        min_value=1E-5,
                                        max_value=1E-2,
                                        default=1E-5,
                                        step=10,
                                        sampling='log')
                feature_selection_regularizer = tf.keras.regularizers.L1(penalty)
            else:
                feature_selection_regularizer = None

        # Text vectorization
        precompiled_vocabulary_file = None if preprocessed_dataset else self._vocabulary_file_path
        embedding_dimensions = hparams.Int('embedding-dimensions',
                                           min_value=1,
                                           max_value=20,
                                           default=5,
                                           step=1)
        embeddings_layer: EmbeddingsReductionLayer = \
            EmbeddingsReductionLayer(precompiled_vocabulary_file=precompiled_vocabulary_file,
                                     embedding_dimensions=embedding_dimensions,
                                     embeddings_regularizer=feature_selection_regularizer)
        if preprocessed_dataset:
            embeddings_layer.adapt(dataset=preprocessed_dataset)
        _LOGGER.info(f'Vocabulary length: {len(embeddings_layer.vocabulary)} words')
        _LOGGER.info(embeddings_layer.vocabulary)
        # Store vocabulary to training output dir
        embeddings_layer.save_vocabulary_to_file(file_path=os.path.join(self._train_log_dir, 'vocabulary.txt'))
        # Lookup embeddings and perform word reduction
        embeddings = embeddings_layer(input_text_preprocessed)  # -> [bdim, n_features, n_words=1, n_embeddings]

        # Numerical preprocessing
        numerical_normalisation_layer = InstanceNormalisationLayer(axis=-1,
                                                                   name='NumericalNormalisation')
        # Make sure InstanceNormalisationLayer.build() is implicitly called before (potentially) loading weights.
        # If not, number of internal weights are zero.
        input_numerical_normalized = numerical_normalisation_layer(input_numerical)
        if self._datasets.dataset_train_numerical:
            _LOGGER.info('Adapting normalisation layer from dataset')
            numerical_normalisation_layer.adapt_from_dataset(data=self._datasets.dataset_train_numerical)
        else:
            _LOGGER.info(f'Loading normalisation weights from file {self._numerical_normalisation_weights_file_path}')
            numerical_normalisation_layer.load_saved_weights_file(file_path=self._numerical_normalisation_weights_file_path)
        # Store normalisation weights to training output dir
        normalisation_weights_file_path: str = os.path.join(self._train_log_dir, 'normalisation.tar')
        numerical_normalisation_layer.save_weights_to_file(file_path=normalisation_weights_file_path)
        _LOGGER.info(f'Saved normalisation weights to {normalisation_weights_file_path}')

        # Flatten word vector to -> [bdim, n_features * n_embeddings]
        embeddings_flat = tf.reshape(embeddings, (-1, len(self._features_text) * embedding_dimensions))

        # Normalization of numerical features (per feature channel)
        # No need to normalize the embeddings since they're nicely distributed
        # Concatenate word vector and numerical features -> [bdim, n_text * n_embeddings + n_numerical]
        branch_dense_0 = hparams.Int('branch_dense_0',
                                     min_value=32,
                                     max_value=256,
                                     step=32,
                                     default=160)
        branch_dense_1 = hparams.Int('branch_dense_1',
                                     min_value=32,
                                     max_value=256,
                                     step=32,
                                     default=224)
        embeddings_branch = tf.keras.layers.Dense(units=branch_dense_0,
                                                  activation='relu',
                                                  kernel_regularizer=None)(embeddings_flat)
        embeddings_branch = tf.keras.layers.Dense(units=branch_dense_1,
                                                  activation='relu',
                                                  kernel_regularizer=None)(embeddings_branch)
        numerical_branch = tf.keras.layers.Dense(units=branch_dense_0,
                                                 activation='relu',
                                                 kernel_regularizer=feature_selection_regularizer)(input_numerical_normalized)
        numerical_branch = tf.keras.layers.Dense(units=branch_dense_1,
                                                 activation='relu',
                                                 kernel_regularizer=None)(numerical_branch)
        complete_feature_vector = tf.keras.layers.Concatenate(axis=1, name='ConcatFeatures')([embeddings_branch,
                                                                                              numerical_branch])
        _LOGGER.info(f'Feature vector shape {complete_feature_vector.get_shape()}')

        # Autoencoder dense layer
        with_feature_multicollinearity_regularizer = hparams.Boolean('feature_multicollinearity_regularisation',
                                                                     default=False)
        with hparams.conditional_scope('feature_multicollinearity_regularisation', [True]):
            if with_feature_multicollinearity_regularizer:
                # L2; regularisation to deal with multicollinearity
                correlation_penalty = hparams.Float('feature_multicollinearity_regularisation_penalty',
                                                    min_value=1E-9,
                                                    max_value=1E-2,
                                                    default=1E-9,
                                                    step=10,
                                                    sampling='log')
                regularizer = tf.keras.regularizers.L2(correlation_penalty)
            else:
                regularizer = None
        activation = hparams.Choice('dense-activation',
                                    values=['relu', 'sigmoid', 'linear'],
                                    default='relu')
        _LOGGER.info(f'length feature vector {len(self._features)}')
        layers: int = hparams.Int('dense-layers',
                                  min_value=1,
                                  max_value=6,
                                  default=3,
                                  step=1)
        units: int = hparams.Int('dense-units',
                                 min_value=32,
                                 max_value=1024,
                                 default=608,
                                 step=32)
        delta_factor: float = hparams.Float('dense-units-reduction',
                                            min_value=0.1,
                                            max_value=0.2,  # Must match 1 / max(n_layers - 1)
                                            default=0.14,
                                            step=0.01)
        dropout_rate = hparams.Float(name='dropout_rate',
                                     min_value=0,
                                     max_value=0.9,
                                     step=0.1,
                                     default=0.2)
        x = complete_feature_vector
        for layer_idx in range(0, layers):
            x = tf.keras.layers.Dense(units=units - (layer_idx * int(np.floor(delta_factor * units))),
                                      activation=activation,
                                      kernel_regularizer=regularizer)(x)    # -> [bdim, n_units]
            if dropout_rate >= 0:
                x = tf.keras.layers.Dropout(rate=dropout_rate,
                                            seed=1)(x)


        # Specify network out shape
        # Set initial bias in the model output layer to bias the model to expected TP/TN skew
        bias_initializer = tf.keras.initializers.Constant(self._datasets.model_bias_estimate)
        confidences = tf.keras.layers.Dense(units=1,
                                            bias_initializer=bias_initializer,
                                            name='Confidences',
                                            activation='sigmoid')(x)  # -> [bdim, 1]

        def _get_input_tensor_with_name(name: str) -> Union[tf.Tensor, tf.RaggedTensor]:
            # Helper to assemble input tensor in order defined by self._features
            all_input_tensors = text_inputs.copy()
            all_input_tensors.extend(numerical_inputs)
            for input_tensor in all_input_tensors:
                if input_tensor.name == name:
                    return input_tensor
            raise ValueError(f'Found no input tensor with name {name}')
        model_inputs = []  # Flat list of model inputs [feature0, feature1, ... ]
        for feature_name in self._features:
            model_inputs.append(_get_input_tensor_with_name(name=feature_name))
        self._keras_model = KerasCustomMetricModel(inputs=model_inputs,
                                                   outputs=confidences,
                                                   metric_specs=CUSTOM_METRICS)

        metrics = [tf.keras.metrics.TruePositives(),
                   tf.keras.metrics.TrueNegatives(),
                   tf.keras.metrics.FalsePositives(),
                   tf.keras.metrics.FalseNegatives(),
                   tf.keras.metrics.BinaryAccuracy(),
                   tf.keras.metrics.AUC(),
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall(),
                   MccScore(),
                   F1Score()]
        optimizer_algo = hparams.Fixed('optimizer', 'Adam')
        # TODO: Rework this snippet using tf.keras.optimizers.get() with custom kwargs (buggy)
        if optimizer_algo == 'Adam':
            optimizer_cls = tf.keras.optimizers.Adam
        elif optimizer_algo == 'Adadelta':
            optimizer_cls = tf.keras.optimizers.Adadelta
        else:
            raise ValueError(f'Undefined optimizer: {optimizer_algo}')
        optimizer = optimizer_cls(learning_rate=hparams.Float('learning-rate',
                                                              min_value=1E-5,
                                                              max_value=1E-3,
                                                              default=1E-5,
                                                              step=10,
                                                              sampling='log'))

        def loss_wrapper(x, y, y_pred, sample_weight):
            """
            Wrapper function to inspect loss function.
            """
            y, = y  # unpack tuple
            # Uncomment below to view input data, labels and predictions in raw format
            #x = print_tensor_op(x, 'x', 5)
            #y_pred = print_tensor_op(y_pred, 'y_pred')
            #y = print_tensor_op(y, 'y')
            return self._keras_model.default_loss(x, y, y_pred, sample_weight)

        self._keras_model.default_loss = self._keras_model.compute_loss  # Save loss computation method as default_loss
        self._keras_model.compute_loss = loss_wrapper  # Replace model loss computation with wrapper
        self._keras_model.compile(optimizer=optimizer,
                                  loss=self.loss_fn,
                                  metrics=metrics)
        self._keras_model.summary(line_length=160)

        return self._keras_model

    @staticmethod
    @tf.keras.saving.register_keras_serializable()
    def loss_fn(y_true, y_pred) -> tf.Tensor:
        c = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
        return c

    def save_model_fn(self, epoch: int, logs=Optional[dict]):
        """
        Saves model to Keras saved model format
        :param epoch: Current epoch
        :param logs: Dictionary of batch statistics
        """
        _LOGGER.info(epoch, logs)
        # NOTE: The suffix .keras is important to tf.keras.saving.load_model()
        filepath = self._train_log_dir + '/saved-models/%d-%.4f.keras' % (epoch, logs['val_loss'])
        _LOGGER.info(f'Saving model to {filepath}')
        self._keras_model.save(filepath=filepath)

    def _generate_dataset_tensor_signature(self) -> Tuple[Tuple[tf.TensorSpec, ...], ...]:
        """
        Helper method to generatate HD5 -> TF data generator tensor signatures.
        :return: A nested tuple of tf.TensorSpec instances
        """
        def _get_input_signature_from_name(name: str) -> Union[tf.TensorSpec, tf.RaggedTensorSpec]:
            # Helper to assemble input tensor in order defined by self._features
            if name in self._features_text:
                return tf.TensorSpec((), dtype=tf.string, name=name)  # (, 1)
            elif name in self._features_numerical:
                return tf.TensorSpec((), dtype=tf.float32, name=name)  # (, 1)
            else:
                raise ValueError(f'Found no input tensor with name {name}')

        input_tensor_signatures = ()
        for feature_name in self._features:
            input_tensor_signatures += (_get_input_signature_from_name(name=feature_name), )
        signature = (
            input_tensor_signatures,
            (tf.TensorSpec((1, ), dtype=tf.float32, name='label'), )
        )
        return signature

    def _init_datasets(self,
                       hd5_file_path: str,
                       hparams: HyperParameters,
                       compile_vocabulary_normalisation_factors: bool = True,
                       init_test_data: bool = True) -> InitializedDatasets:

        batch_size: int = hparams.Int('batch_size',
                                      min_value=64,
                                      max_value=256,
                                      step=32,
                                      default=64)

        # Training setup
        hd5_data_generator_train: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                      group_name='train',
                                                                      output_tensor_format=self._features,
                                                                      label='label',
                                                                      expand_1d_categorical_to_2d=False)

        dataset_train: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_train,
                                                                                output_signature=self._generate_dataset_tensor_signature())

        # Training weights
        n_pathogenic, n_benign = hd5_data_generator_train.count_positive_negative_categorical_labels()
        assert n_pathogenic > 0, n_pathogenic
        assert n_benign > 0, n_benign
        _LOGGER.info(f'nTP:{n_pathogenic} ({100 * n_pathogenic / hd5_data_generator_train.data_length:.4f}%) \
        , nTN:{n_benign}, n_samples:{hd5_data_generator_train.data_length}')

        # Compute expected model bias based on training data skew, bias = TPs / TNs
        model_bias_estimate: float = np.log(float(n_pathogenic) / float(n_benign))

        @tf.function
        def add_weights(data, labels, **kwargs):
            """
            Helper function to compute weights for balanced loss
            """
            weight_pathogenic = kwargs.get('weight_pathogenic')
            weight_benign = kwargs.get('weight_benign')
            weights = tf.where(condition=tf.equal(labels, tf.constant(LABEL_PATHOGENIC_VARIANT)),
                               x=tf.ones_like(labels) * tf.constant(weight_pathogenic),  # cond == True
                               y=tf.ones_like(labels) * tf.constant(weight_benign))  # cond == False
            return data, labels, weights

        training_weights = hparams.Choice('training_weights',
                                          values=[False, True],
                                          default=False)
        if training_weights:
            # Setup class weights so that dataset is perfectly balanced w.r.t class-ratio-loss imbalance
            weight_pathogenic = (1.0 / float(n_pathogenic)) * (float(hd5_data_generator_train.data_length) / 2.0)
            weight_benign = (1.0 / float(n_benign)) * (float(hd5_data_generator_train.data_length) / 2.0)
            _LOGGER.info(f'class weights: benign:{weight_benign}, pathogenic:{weight_pathogenic}')
            assert weight_pathogenic >= weight_benign
            dataset_train = dataset_train.map(map_func=lambda *args: add_weights(*args,
                                                                                 weight_pathogenic=weight_pathogenic,
                                                                                 weight_benign=weight_benign),
                                              num_parallel_calls=tf.data.AUTOTUNE)

        variant_category_weights = hparams.Choice('variant_weights',
                                                  values=[False, True],
                                                  default=True)  # FIXME
        if training_weights and variant_category_weights:
            raise NotImplementedError('Enabling both training weights and variant category weights not supported.')

        @tf.function
        def add_weights_per_variant_category(data, labels, **kwargs):
            """
            Helper function to compute weights for customising loss per variant category

            Setup 1.0 weight for all samples as default.

            Iterate through the consequence-to-weight mapping and multiply existing weight with scale.
            There might exist multiple annotations per variant, so weights might accumulate
            for a particular variant.
            """
            regexp_category_weights = kwargs.get('regexp_category_weights')
            csq_consequence_tensor = data[kwargs.get('csq_consequence_tensor_idx')]
            all_sample_weights = tf.ones_like(labels)  # Default weight is 1.0 (neutral)
            for i, (regexp, weight) in enumerate(regexp_category_weights.items()):
                regexp_matches = tf.strings.regex_full_match(input=csq_consequence_tensor,
                                                             pattern=regexp,
                                                             name=f'variant_category_weight_{i}_regexp')
                all_sample_weights = tf.where(condition=regexp_matches,
                                              x=all_sample_weights * weight,  # cond == True
                                              y=all_sample_weights)  # cond == False
            return data, labels, all_sample_weights

        if variant_category_weights:
            # intron_variant is ignored
            vep_variant_weighted_categories = \
            ['missense_variant', 'downstream_gene_variant', 'upstream_gene_variant',
             'non_coding_transcript_exon_variant',
             'splice_donor_variant', 'splice_donor_region_variant',
             'splice_region_variant', '5_prime_UTR_variant',
             'splice_polypyrimidine_tract_variant', '3_prime_UTR_variant',
             'synonymous_variant', 'frameshift_variant', 'stop_gained',
             'splice_acceptor_variant', 'splice_donor_5th_base_variant', 'stop_lost',
             'protein_altering_variant', 'inframe_insertion', 'inframe_deletion',
             'transcript_ablation', 'start_lost', 'stop_retained_variant',
             'coding_sequence_variant', 'mature_miRNA_variant',
             'incomplete_terminal_codon_variant']
            regexp_category_weights = dict()
            for category in vep_variant_weighted_categories:
                regexp_category_weights.update({f'.*({category}).*': tf.constant(5.0, dtype=tf.float32)})
            model_input_spec = self._generate_dataset_tensor_signature()
            model_input_data_spec, _ = model_input_spec  # Drop labels
            csq_consequence_tensor_idx = None
            for idx, input_spec in enumerate(model_input_data_spec):
                if input_spec.name == 'most_severe_consequence':
                    csq_consequence_tensor_idx = idx
            assert csq_consequence_tensor_idx is not None
            dataset_train = dataset_train.map(map_func=lambda *args: add_weights_per_variant_category(*args,
                                                                                 regexp_category_weights=regexp_category_weights,
                                                                                 csq_consequence_tensor_idx=csq_consequence_tensor_idx),
                                              num_parallel_calls=tf.data.AUTOTUNE)

        dataset_train = dataset_train.cache()
        dataset_train = dataset_train.repeat(-1)

        # Annotation augmentation for generating novel/ undocumented variants (annotation dropout)
        feature_dropout_ratio = hparams.Choice('feature_dropout_ratio',
                                               values=[0.0, float(1E-3), float(1E-2), 0.5],
                                               default=0.5)
        if feature_dropout_ratio > 0:
            clinvar_clnrevstat_novelizer = TextAugmentDropoutDataset(target_data_tensor_idx=2,
                                                                     dropout_on_categorical_label_value=LABEL_PATHOGENIC_VARIANT,
                                                                     seed=1,
                                                                     dropout_ratio=feature_dropout_ratio)
            dataset_train = clinvar_clnrevstat_novelizer(dataset_train)
            clinvar_clnsig_novelizer = TextAugmentDropoutDataset(target_data_tensor_idx=3,
                                                                 dropout_on_categorical_label_value=LABEL_PATHOGENIC_VARIANT,
                                                                 seed=2,
                                                                 dropout_ratio=feature_dropout_ratio)
            dataset_train = clinvar_clnsig_novelizer(dataset_train)

        # Training occurrence sampling
        expected_amount_of_variants_in_case = float(3.5E6)
        likelihood_pathogenic = 1.0 / expected_amount_of_variants_in_case
        training_occurrence_frq_sampling = hparams.Choice('training_occurrence_frq_sampling',
                                                          values=[True, False],
                                                          default=False)

        @tf.function
        def filt_fn(*args, **kwargs):
            """
            Helper function to filter data samples on label
            """
            if len(args) == 2:
                data, labels = args
            elif len(args) == 3:
                data, labels, weights = args
            else:
                raise ValueError(f'Unknown args: {args}')
            del data
            del weights
            target_label = kwargs.get('target_label')
            predicate = tf.equal(labels, target_label)[0, 0]
            return predicate

        if training_occurrence_frq_sampling:
            _LOGGER.info(f'Sampling pathogenic variants during training with likelihood of {likelihood_pathogenic}')
            sampling_weights = (1.0 - likelihood_pathogenic, likelihood_pathogenic)
            _LOGGER.info(f'Sampling weights (benign, pathogenic): {sampling_weights}')
            train_pathogenic_variants = \
                dataset_train.filter(predicate=lambda *args: filt_fn(*args, target_label=LABEL_PATHOGENIC_VARIANT))
            train_benign_variants = \
                dataset_train.filter(predicate=lambda *args: filt_fn(*args, target_label=LABEL_BENIGN_VARIANT))
            dataset_train = tf.data.Dataset.sample_from_datasets(datasets=(train_benign_variants, train_pathogenic_variants),
                                                                 weights=sampling_weights,
                                                                 seed=1)
            # Setup new bias estimate, since training data is now skewed
            model_bias_estimate = np.log(likelihood_pathogenic)

        dataset_train = dataset_train.shuffle(buffer_size=int(5E5),
                                              seed=1)  # FIXME: Seed

        dataset_train = dataset_train.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset_train = dataset_train.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Vocabulary and normalisation setup
        dataset_vocabulary: tf.data.Dataset = None
        dataset_numerical: tf.data.Dataset = None
        if compile_vocabulary_normalisation_factors:
            _LOGGER.info('Compiling new vocabulary and normalisation factors')
            # Text preprocessing
            hd5_data_generator_vocabulary: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                               group_name='train',
                                                                               output_tensor_format=[self._features_text])
            input_signature_vocabulary: Tuple[tf.TensorSpec] = \
                (tf.TensorSpec((len(self._features_text),), dtype=tf.string, name='input_text_vocabulary'),)
            dataset_vocabulary = get_tf_dataset_from_hd5_data_generator(
                hd5_data_generator=hd5_data_generator_vocabulary,
                output_signature=input_signature_vocabulary)
            dataset_vocabulary = dataset_vocabulary.prefetch(buffer_size=1024)
            # Numerical preprocessing
            hd5_data_generator_numerical = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            group_name='train',
                                                            output_tensor_format=self._features_numerical)
            input_signature_numerical_normalisation = \
                tf.TensorSpec((len(self._features_numerical),), dtype=tf.float32, name='input_numerical_normalisation')
            dataset_numerical: tf.data.Dataset = \
                get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_numerical,
                                                       output_signature=input_signature_numerical_normalisation)

        # Testing setup
        if init_test_data:
            hd5_data_generator_test: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                         group_name='test',
                                                                         output_tensor_format=self._features,
                                                                         label='label',
                                                                         expand_1d_categorical_to_2d=False)
            dataset_test: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_test,
                                                                                   output_signature=self._generate_dataset_tensor_signature())
            dataset_test = dataset_test.cache()
            dataset_test = dataset_test.repeat(-1)
            dataset_test = dataset_test.shuffle(buffer_size=int(5E5),
                                                seed=1)  # FIXME: Seed
            dataset_test = dataset_test.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_test = dataset_test.prefetch(buffer_size=tf.data.AUTOTUNE)
            dataset_test_length = hd5_data_generator_test.data_length
        else:
            dataset_test = None
            dataset_test_length = None

        self._datasets = InitializedDatasets(dataset_train_numerical=dataset_numerical,
                                             dataset_train_vocabulary=dataset_vocabulary,
                                             dataset_train=dataset_train,
                                             dataset_test=dataset_test,
                                             train_data_length=hd5_data_generator_train.data_length,
                                             test_data_length=dataset_test_length,
                                             batch_size=batch_size,
                                             model_bias_estimate=model_bias_estimate)
        _LOGGER.info(f'Datasets init complete: {self._datasets}')

    @staticmethod
    def get_uninitialized_hyperparameters() -> HyperParameters:
        """
        Uninitialized (empty) hparams confers hparam configuration to the model build step.
        """
        return HyperParameters()

    def build(self,
              hd5_file_path: str,
              hparams: HyperParameters,
              compile_vocabulary_normalisation_factors: bool,
              train_log_dir_already_exist: bool = False) -> tf.keras.Model:
        """
        Main method to initialize datasets and build model based on hyperparameter config.
        :param hd5_file_path: The HDF5 file path used for training, test
        :param hparams: hyperparameter config (new empty instance or as created by hyperparameter tuner)
          Supplying a new instance of Hyperparameters creates a model with default hyperparam configs.
        :param compile_vocabulary_normalisation_factors: Compile new vocabulary and normalisation factors from data
        :param train_log_dir_already_exist: Reuse existing directory for this build-training run
        :return: built keras model
        """
        # Set up a directory containing model build and training output
        os.makedirs(self._train_log_dir, exist_ok=train_log_dir_already_exist)
        self._init_datasets(hd5_file_path=hd5_file_path,
                            hparams=hparams,
                            compile_vocabulary_normalisation_factors=compile_vocabulary_normalisation_factors)
        self._build_model(hparams=hparams)
        _LOGGER.info(f'Hyperparameters: {hparams.values}')
        with open(os.path.join(self._train_log_dir, 'hyperparams.txt'), 'w') as file:
            for key, value in hparams.values.items():
                file.write(f'{key}={value}\n')
        return self._keras_model

    def train(self,
              hparam_tuning_callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.callbacks.History:
        """
        Execute training and evaluation of pre built model.
        :param hparam_tuning_callbacks: List of keras tuner callbacks to be appended to fit() call
        :return: A History object containing the training progress
        """

        if self._datasets is None:
            raise ValueError('Expected initialized datasets, but got None')

        steps_per_epoch = int(np.ceil(float(self._datasets.train_data_length) / float(self._datasets.batch_size)))

        # Setup logging and store configuration files
        callbacks: List[tf.keras.callbacks.Callback] = list()
        if hparam_tuning_callbacks is not None:
            callbacks.extend(hparam_tuning_callbacks)
        if 'TensorBoard' in [cb.__class__ for cb in callbacks]:
            pass
        else:
            # Setup default monitoring in Tensorboard
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self._train_log_dir,
                                                            histogram_freq=1,
                                                            embeddings_freq=1))  # FIXME: Bug in Keras
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=self.save_model_fn))
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          verbose=1,
                                                          patience=3))
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())

        compile_config: Dict[str, Any] = self._keras_model.get_compile_config()
        network_config: str = self._keras_model.to_json()
        with open(os.path.join(self._train_log_dir, 'compile-config.txt'), 'w') as file:
            file.write(dumps(compile_config))
        with open(os.path.join(self._train_log_dir, 'network-config.txt'), 'w') as file:
            file.write(network_config)
        with open(os.path.join(self._train_log_dir, 'dataset-config.txt'), 'w') as file:
            file.write('Dataset train:\n' + str(vars(self._datasets.dataset_train)))
            file.write('Dataset test:\n' + str(vars(self._datasets.dataset_test)))
        with open(os.path.join(self._train_log_dir, 'build-config.txt'), 'w') as file:
            file.write(str(globals()))
            file.write(str(locals()))

        validation_steps = int(np.ceil(float(self._datasets.test_data_length) / float(self._datasets.batch_size)))

        history = self._keras_model.fit(x=self._datasets.dataset_train,
                                        batch_size=1,
                                        epochs=int(1E2),
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=self._datasets.dataset_test,
                                        validation_steps=validation_steps,
                                        callbacks=callbacks,
                                        verbose=2)
        return history

    def train_model_explainer(self):
        """
        Setup model explainer on training data, to provide model inference explanations
        during runtime.

        This method expects that a previously trained self._keras_model is available.

        WARNING: This method exports training data to file, and in case of
        patient data in the training set, this will "leak" sensitive data.
        On calling this method, make sure training data set does not contain
        sensitive data, or manage the output file appropriately.
        """
        _LOGGER.info('Training model explainer')
        if self._keras_model is None:
            raise ValueError(f'No available keras model to compute predictions for')
        if self._datasets is None:
            raise ValueError(f'No datasets available')
        # NOTE: Changes to below configuration must be reflected in the self._load_saved_model_explainer()
        model_input_spec = self._generate_dataset_tensor_signature()
        data_tensor_spec, _ = model_input_spec  # Drop labels spec
        self._model_explainer = ModelExplainer(model=self._infer_pathogenicity_scores,
                                               input_tensor_spec=data_tensor_spec)
        dataset = self._datasets.dataset_train
        # FIXME: The data used for fitting the explainer should be randomly selected from the complete set of training data
        self._model_explainer.adapt(dataset=dataset)
        gc.collect()
        file_path = os.path.join(self._train_log_dir, 'model-explainer.bin')  # Might contain sensitive data!
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self._model_explainer.save(file_path=file_path)
        _LOGGER.info(f'Saved model explainer to {file_path}')

    def post_train_explainer(self, model_path: str, hd5_file_path: str):
        """
        Post train a model explain from saved keras model.
        """
        self._load_saved_keras_model(model_path=model_path)
        self._init_datasets(hd5_file_path=hd5_file_path,
                            hparams=HyperParameters())
        self.train_model_explainer()

    def _load_saved_keras_model(self, model_path: str,):
        """
        Load trained model from model_path
        :param model_path: Path to Keras saved model (*.keras) zip file
        """
        # Load Keras Model
        if self._keras_model is not None:
            raise ValueError('The model already contains a loaded keras model!')
        model = tf.keras.saving.load_model(model_path)
        _LOGGER.info(f'Model input: {model.inputs}')
        _LOGGER.info(f'Model output: {model.outputs}')
        model.summary(line_length=160)
        self._keras_model = model

    def _load_saved_model_explainer(self, model_explainer_path: str):
        """
        Load saved ModelExplainer from path
        :param model_explainer_path: Path to ModelExplainer binary file
        """

        # Load ModelExplainer
        if self._model_explainer is not None:
            raise ValueError('The model already contains a loaded ModelExplainer!')
        _LOGGER.info(f'Loading ModelExplainer from {model_explainer_path}')
        if self._keras_model is None:
            raise ValueError(f'Loading ModelExplainer requires a pre-loaded keras model, none currently loaded')
        # NOTE: Changes to below configuration must be reflected in the self.train_model_explainer()
        model_input_spec = self._generate_dataset_tensor_signature()
        model_input_data_spec, _ = model_input_spec  # Drop labels
        self._model_explainer = ModelExplainer.from_saved_file(file_path=model_explainer_path,
                                                               keras_model=self._infer_pathogenicity_scores,
                                                               input_tensor_spec=model_input_data_spec)

    def load_saved_model(self,
                         keras_model_path: str = DEFAULT_MODEL_SPEC.keras_model,
                         model_explainer_path: str = DEFAULT_MODEL_SPEC.explainer_model):
        """
        Main interface to load a saved instance from file.
        :param keras_model_path: Path to saved keras file (*.keras)
        :param model_explainer_path: Path to saved model explainer instance (model-explainer.bin)
        """
        self._load_saved_keras_model(model_path=keras_model_path)
        self._load_saved_model_explainer(model_explainer_path=model_explainer_path)

    def _infer_pathogenicity_scores(self,
                                    tensor_dict: Dict[str, tf.Tensor]) -> np.ndarray:
        """
        Main method to compute inferences from input tensors.
        :param tensor_dict: Input data tensors as dict, key is the tensor name
        :return: 1D scores same size as outer, batch dimension
        """
        if self._keras_model is None:
            raise ValueError('No keras model available for inference computation!')
        score_classes = self._keras_model(tensor_dict)  # [class benign, class pathogenic]
        score_classes = score_classes.numpy()
        prediction_class_pathogenic = score_classes[:, 0]
        return prediction_class_pathogenic

    def predict_on_hd5(self,
                       hd5_file_path: str,
                       group_names: Set[str] = {'train', 'test'},
                       batch_size: int = 1000) -> str:

        """
        Creates a .hd5 file containing inpute feature data, ground truth and inferences side-by-side.
        :param hd5_file_path: The file to the input data file for creating inferences
        :param group_names: The group names in the hd5 to load data and to compute inferences for
        :param batch_size: Batch size, a large batch size improves speed
        :returns: The path to the .hd5 file containing data and inferences
        """

        # Set up output file
        if not '.hd5' in hd5_file_path:
            raise ValueError('Expected a .hd5 file as input')
        output_file_path = hd5_file_path.replace('.hd5', '-inferences.hd5')
        if output_file_path == hd5_file_path:
            raise ValueError('Won\'t overwrite input data')
        output_file = Hdf5File(name=output_file_path,
                               mode='w')

        for group_name in group_names:
            # TODO: Make sure config to Hd5DataGenerator is identical to train time setup
            datagen: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                         group_name=group_name,
                                                         output_tensor_format=self._features,
                                                         label='label',
                                                         expand_1d_categorical_to_2d=True)
            dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=datagen,
                                                             output_signature=self._generate_dataset_tensor_signature())
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=10)
            data_length = datagen.data_length

            output_group = output_file.create_group(group_name)
            # TODO: Mimic data set settings in ../dataset/dataset.py
            for dataset_name in self._features_numerical:
                output_group.create_dataset(name=dataset_name,
                                            dtype=np.float32,
                                            fillvalue=np.nan,
                                            shape=(data_length, ))
            for dataset_name in self._features_text:
                output_group.create_dataset(name=dataset_name,
                                            dtype=string_dtype(),
                                            fillvalue=b'\0',
                                            shape=(data_length, ))
            output_group.create_dataset(name='ground-truth',
                                        dtype=np.float32,
                                        fillvalue=np.nan,
                                        shape=(data_length, ))
            output_group.create_dataset(name='prediction',
                                        dtype=np.float32,
                                        fillvalue=np.nan,
                                        shape=(data_length, ))

            processed_sample_idx = 0
            model_input_spec = self._generate_dataset_tensor_signature()
            model_input_data_spec, _ = model_input_spec  # Drop labels
            for data, labels in dataset.as_numpy_iterator():
                data: Tuple[tf.Tensor]
                label, = labels
                label_class_pathogenic = label[:, 1]
                input_tensor_dict: Dict[str, tf.Tensor] = {}
                for input_feature_idx, tensor_spec in enumerate(model_input_data_spec):
                    input_tensor_dict.update({
                        tensor_spec.name: tf.constant(data[input_feature_idx], dtype=tensor_spec.dtype, name=tensor_spec.name)
                    })
                prediction_class_pathogenic = self._infer_pathogenicity_scores(tensor_dict=input_tensor_dict)
                for feature_name, tensor in input_tensor_dict.items():
                    output_group[feature_name][processed_sample_idx:processed_sample_idx + batch_size] = tensor.numpy()
                output_group['ground-truth'][processed_sample_idx:processed_sample_idx+batch_size] = label_class_pathogenic
                output_group['prediction'][processed_sample_idx:processed_sample_idx + batch_size] = prediction_class_pathogenic
                processed_sample_idx += batch_size
                _LOGGER.info(f"Progress {group_name}: %.2f%%" % (100.0 * (processed_sample_idx / data_length)))
            output_file.flush()
            gc.collect()
        output_file.close()
        return output_file_path

    def score_variant(self,
                      variants: List[ParsableVariant],
                      explain_variant_score_threshold: float = 0.9) -> pd.DataFrame:
        """
        Run model inference step on ParsableVariant instance.
        :param variants: The variants to score
        :param explain_variant_score_threshold: Explain variant predictions >= this threshold
        :return: Rank scores, (0, 1), the higher the more pathogenic.
          Input-output order is preserved.
        """
        # TODO: store input config in model training step
        # TODO: Sanity check that the VCF file is actually annotated with expected VCF data
        def get_str_feature(variant: ParsableVariant,
                            name: str) -> bytes:
            if name in variant.parsed_fields:
                return variant.__getattribute__(name)
            else:
                return b''

        def get_num_feature(variant: ParsableVariant,
                            name: str) -> float:
            if name in variant.parsed_fields:
                value = variant.__getattribute__(name)
                if isinstance(value, (bytes, str)) and len(value) == 0:
                    pass
                else:
                    return value
            return 0.0

        model_input_spec = self._generate_dataset_tensor_signature()
        model_input_data_spec, _ = model_input_spec  # Drop labels
        input_dict: Dict[str, Union[List, tf.Tensor]] = {}
        for tensor_spec in model_input_data_spec:
            input_dict.update({tensor_spec.name: []})
            for variant in variants:
                if tensor_spec.dtype == tf.string:
                    input_dict[tensor_spec.name].append(get_str_feature(variant=variant, name=tensor_spec.name))
                elif tensor_spec.dtype == tf.float32:
                    input_dict[tensor_spec.name].append(get_num_feature(variant=variant, name=tensor_spec.name))
                else:
                    raise ValueError(f'Unmapped input data dtype spec: {tensor_spec}')
            # Convert to Tensor
            tensor_data = input_dict[tensor_spec.name].copy()
            input_dict[tensor_spec.name] = tf.constant(value=tensor_data,
                                                       dtype=tensor_spec.dtype,
                                                       name=tensor_spec.name)
        pathogenicity_scores = self._infer_pathogenicity_scores(tensor_dict=input_dict)
        if len(pathogenicity_scores) != len(variants):
            raise ValueError(f'Expected same amount of predictions as input data')
        idx_scores_above_threshold = np.flatnonzero(pathogenicity_scores >= explain_variant_score_threshold)
        df_dict = {}
        for tensor_name, tensor in input_dict.items():
            df_dict.update({tensor_name: tensor.numpy()[idx_scores_above_threshold]})
        df_selected_variants_for_explanation = pd.DataFrame.from_dict(df_dict)
        explanations_full = np.empty(shape=(len(variants), len(df_selected_variants_for_explanation.columns)))
        explanations_full.fill(np.nan)
        if len(idx_scores_above_threshold) > 0:
            explanations_full[idx_scores_above_threshold, :] = self._model_explainer.shap_values(X=df_selected_variants_for_explanation.values,
                                                                                                 gc_collect=True)  # FIXME: Method call not supposed to be erroneous by typechecker
        explanations_df = pd.DataFrame(data=explanations_full, columns=self._features)
        result_df = pd.concat(objs=(pd.Series(pathogenicity_scores, name='pathogenicity_score'),
                                    explanations_df),
                              axis=1)
        return result_df

