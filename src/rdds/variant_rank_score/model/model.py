# Make sure to set all seeds
from rdds.lib.determinism import enable_determinism; enable_determinism()

# Add NaN, +-Inf checks
from rdds.lib.tf import enable_check_numerics; enable_check_numerics()

from typing import *
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
from rdds.lib.tf import rejection_resample
from rdds.lib.tf import print_tensor_op
from rdds.lib.hpt import HyperParameters
from rdds.lib.vcf import ParsableVariant



@dataclass
class InitializedDatasets:
    dataset_train: tf.data.Dataset
    dataset_test: tf.data.Dataset
    train_data_length: int
    test_data_length: int
    batch_size: int
    dataset_train_numerical: tf.data.Dataset = None
    dataset_train_vocabulary:  tf.data.Dataset = None


"""
NOTE!
Always make sure to keep a reference to all layers created in the model, i.e.
layer = tf.keras.layers.MyLayer()
y = layer(x)
If this is not the case, the layer might be removed in the graph!
"""

# TODO: Determine whether to use GeneticModels_family_id?
# TODO: Determine whether to use ModelScore_family_id?

# FIXME: IT's required to set this variable to None on first run, to generate a vocabulary. Then restart training using this file.
# See comment in the text_vectorization_layer.py about keras and tensorboard.
_DEFAULT_VOCABULARY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vocabulary.txt'))
_DEFAULT_NUMERICAL_NORMALISATION_WEIGHTS = os.path.abspath(os.path.join(os.path.dirname(__file__), 'normalisation.tar'))
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
                 vocabulary_file_path: str = _DEFAULT_VOCABULARY_FILE,
                 numerical_normalisation_weights_file_path: str = _DEFAULT_NUMERICAL_NORMALISATION_WEIGHTS,
                 workdir: str = WORKDIR,
                 workdir_suffix: str = 'models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")):
        """

        :param features_text:
        :param features_numerical:
        :param vocabulary_file_path:
        :param numerical_normalisation_weights_file_path:
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

    def _build_model(self,
                     hparams: HyperParameters) -> tf.keras.models.Model:
        """
        :param hparams: Hyperparameters for the model
        """
        input_text: tf.keras.Input = tf.keras.Input(shape=len(self._features_text),
                                                    ragged=True,
                                                    dtype=tf.string,
                                                    name='input_text')
        input_numerical: tf.keras.Input = tf.keras.Input(shape=len(self._features_numerical),
                                                         dtype=tf.float32,
                                                         name='input_numerical')
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
                                           min_value=18,
                                           max_value=40,
                                           default=23,
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
                                     default=64)
        branch_dense_1 = hparams.Int('branch_dense_1',
                                     min_value=32,
                                     max_value=128,
                                     step=32,
                                     default=96)
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
                                                                     default=True)
        with hparams.conditional_scope('feature_multicollinearity_regularisation', [True]):
            if with_feature_multicollinearity_regularizer:
                # L2; regularisation to deal with multicollinearity
                correlation_penalty = hparams.Float('feature_multicollinearity_regularisation_penalty',
                                                    min_value=1E-10,
                                                    max_value=1E-7,
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
                                  max_value=3,
                                  default=2,
                                  step=1)
        units: int = hparams.Int('dense-units',
                                 min_value=32,
                                 max_value=400,
                                 default=224,
                                 step=32)
        delta_factor: float = hparams.Float('dense-units-reduction',
                                            min_value=0.1,
                                            max_value=0.5,  # Must match 1 / max(n_layers - 1)
                                            default=0.28,
                                            step=0.01)
        dropout_rate = hparams.Float(name='dropout_rate',
                                     min_value=0,
                                     max_value=0.9,
                                     step=0.1,
                                     default=0.4)
        x = complete_feature_vector
        for layer_idx in range(0, layers):
            x = tf.keras.layers.Dense(units=units - (layer_idx * int(np.floor(delta_factor * units))),
                                      activation=activation,
                                      kernel_regularizer=regularizer)(x)    # -> [bdim, n_units]
            if dropout_rate >= 0:
                x = tf.keras.layers.Dropout(rate=dropout_rate,
                                            seed=1)(x)


        # Specify network out shape
        logits = tf.keras.layers.Dense(units=2, name='Logits', activation='linear')(x)  # -> [bdim, 2]

        # Softmax layer
        confidences = tf.keras.layers.Softmax(name='Confidences')(logits)  # -> [bdim, 2]

        self._keras_model = tf.keras.Model(inputs=[input_text, input_numerical],
                                           outputs=confidences)

        metrics = [tf.keras.metrics.TruePositives(),
                   tf.keras.metrics.TrueNegatives(),
                   tf.keras.metrics.FalsePositives(),
                   tf.keras.metrics.FalseNegatives(),
                   tf.keras.metrics.CategoricalAccuracy(),
                   tf.keras.metrics.AUC(),
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall()]
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
                                                              default=1E-4,
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
        c = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
        return c

    @staticmethod
    def count_feature_types(hd5_output_dtypes: Dict[str, Type]) -> Tuple[int, int]:
        """
        Computes amount of numerical vs textbased features in dataset.
        :param hd5_output_dtypes: The output types from hd5_data_generator.data_types attribute
        :return: Tuple of count
        """
        n_text_features = 0
        n_numerical_features = 0
        for feature_name, feature_dtype in hd5_output_dtypes.items():
            if feature_dtype == bytes:
                n_text_features += 1
            elif feature_dtype == float:
                n_numerical_features += 1
            else:
                raise ValueError('Unknown feature dtype', feature_dtype)
        return n_text_features, n_numerical_features

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

    def _init_datasets(self,
                       hd5_file_path: str,
                       hparams: HyperParameters,
                       compile_vocabulary_normalisation_factors: bool = True) -> InitializedDatasets:

        batch_size: int = hparams.Int('batch_size',
                                      min_value=128,
                                      max_value=1024,
                                      step=128,
                                      default=256)

        # Training setup
        hd5_data_generator_train: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                      group_name='train',
                                                                      output_tensor_format=[self._features_text,
                                                                                            self._features_numerical],
                                                                      label='label')
        n_text_features, n_numerical_features = self.count_feature_types(hd5_output_dtypes=hd5_data_generator_train.data_types)
        input_signature = ((tf.TensorSpec((n_text_features, ), dtype=tf.string),
                           tf.TensorSpec((n_numerical_features, ), dtype=tf.float32, name='input_numerical')),
                           (tf.TensorSpec((2, ), dtype=tf.float32, name='label'), ))
        dataset_train: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_train,
                                                                                output_signature=input_signature)
        dataset_train = dataset_train.cache()
        dataset_train = dataset_train.repeat(-1)
        dataset_train = dataset_train.shuffle(buffer_size=int(5E5),
                                              seed=1)  # FIXME: Seed
        #dataset_train = rejection_resample(dataset=dataset_train,
        #                                   desired_class_ratio=[0.5, 0.5],
        #                                   seed=1)
        dataset_train = dataset_train.batch(batch_size)
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
                (tf.TensorSpec((n_text_features,), dtype=tf.string, name='input_text_vocabulary'),)
            dataset_vocabulary = get_tf_dataset_from_hd5_data_generator(
                hd5_data_generator=hd5_data_generator_vocabulary,
                output_signature=input_signature_vocabulary)
            dataset_vocabulary = dataset_vocabulary.prefetch(buffer_size=1024)
            # Numerical preprocessing
            hd5_data_generator_numerical = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                            group_name='train',
                                                            output_tensor_format=self._features_numerical)
            input_signature_numerical_normalisation = \
                tf.TensorSpec((n_numerical_features,), dtype=tf.float32, name='input_numerical_normalisation')
            dataset_numerical: tf.data.Dataset = \
                get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_numerical,
                                                       output_signature=input_signature_numerical_normalisation)

        # Testing setup
        hd5_data_generator_test: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                     group_name='test',
                                                                     output_tensor_format=[self._features_text,
                                                                                           self._features_numerical],
                                                                     label='label')
        dataset_test: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_test,
                                                                               output_signature=input_signature)
        dataset_test = dataset_test.cache()
        dataset_test = dataset_test.repeat(-1)
        dataset_test = dataset_test.shuffle(buffer_size=int(5E5),
                                            seed=1)  # FIXME: Seed
        #dataset_test = rejection_resample(dataset=dataset_test,
        #                                  desired_class_ratio=[0.5, 0.5],
        #                                  seed=1)
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.prefetch(buffer_size=tf.data.AUTOTUNE)
        _LOGGER.info(f'Model Input data mapping: {input_signature}')

        self._datasets = InitializedDatasets(dataset_train_numerical=dataset_numerical,
                                             dataset_train_vocabulary=dataset_vocabulary,
                                             dataset_train=dataset_train,
                                             dataset_test=dataset_test,
                                             train_data_length=hd5_data_generator_train.data_length,
                                             test_data_length=hd5_data_generator_test.data_length,
                                             batch_size=batch_size)
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
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          mode='min',
                                                          verbose=1,
                                                          patience=10))
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
                                        epochs=int(3E2),
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=self._datasets.dataset_test,
                                        validation_steps=validation_steps,
                                        callbacks=callbacks,
                                        verbose=2)
        return history

    def load_saved_model(self, model_path: str):
        """
        Load trained model from model_path
        :param model_path: Path to Keras saved model (*.keras) zip file
        """
        if self._keras_model is not None:
            raise ValueError('The model already contains a loaded keras model!')
        model = tf.keras.saving.load_model(model_path)
        _LOGGER.info(f'Model input: {model.inputs}')
        _LOGGER.info(f'Model output: {model.outputs}')
        model.summary(line_length=160)
        self._keras_model = model

    def predict(self, input_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run predict, inference call on input data dictionary.
        :param input_data: Dictionary of input data. Input shapes, data order
          should conform to earlier established input data format.
        :return: Inferences
        """
        return self._keras_model.predict(x=input_data)

    def predict_on_hd5(self,
                       hd5_file_path: str,
                       group_names: Set[str] = {'train', 'test'}) -> str:

        """
        Creates a .hd5 file containing inpute feature data, ground truth and inferences side-by-side.
        :param hd5_file_path: The file to the input data file for creating inferences
        :param group_names: The group names in the hd5 to load data and to compute inferences for
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
            output_tensor_format = [self._features_text, self._features_numerical]
            datagen: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                         group_name=group_name,
                                                         output_tensor_format=output_tensor_format,
                                                         label='label',
                                                         expand_1d_categorical_to_2d=True)
            n_text_features, n_numerical_features = \
                self.count_feature_types(hd5_output_dtypes=datagen.data_types)
            input_signature = ((tf.TensorSpec((n_text_features, ), dtype=tf.string),
                               tf.TensorSpec((n_numerical_features, ), dtype=tf.float32, name='input_numerical')),
                               (tf.TensorSpec((2, ), dtype=tf.float32, name='label'), ))
            dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=datagen,
                                                             output_signature=input_signature)
            dataset = dataset.batch(10000)
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
            for data, labels in dataset.as_numpy_iterator():
                label, = labels
                label_class_pathogenic = label[:, 1]
                tensor_str, tensor_numerical = data
                r = self._keras_model([tensor_str, tensor_numerical])
                r = r.numpy()
                prediction_class_pathogenic = r[:, 1]
                batch_size = tensor_str.shape[0]
                for feature_names, features_data in [(self._features_text, tensor_str),
                                                     (self._features_numerical, tensor_numerical)]:
                    for feature_idx, feature_name in enumerate(feature_names):
                        output_group[feature_name][processed_sample_idx:processed_sample_idx+batch_size] = features_data[:, feature_idx]
                output_group['ground-truth'][processed_sample_idx:processed_sample_idx+batch_size] = label_class_pathogenic
                output_group['prediction'][processed_sample_idx:processed_sample_idx + batch_size] = prediction_class_pathogenic
                processed_sample_idx += batch_size
                _LOGGER.info(f"Progress {group_name}: %.2f%%" % (100.0 * (processed_sample_idx / data_length)))
            output_file.flush()
            gc.collect()
        output_file.close()
        return output_file_path

    def score_variant(self, variants: List[ParsableVariant]) -> List[float]:
        """
        Run model inference step on ParsableVariant instance.
        :param variants: The variants to score
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

        text_features_batch: List[List[bytes]] = []
        numerical_features_batch: List[List[float]] = []
        for variant in variants:
            text_features = [get_str_feature(variant, feature_name) for feature_name in self._features_text]
            text_features_batch.append(text_features)
            numerical_features = [get_num_feature(variant, feature_name) for feature_name in self._features_numerical]
            numerical_features_batch.append(numerical_features)
        tensor_text = tf.constant(text_features_batch, dtype=tf.string)
        tensor_numerical = tf.constant(numerical_features_batch, dtype=tf.float32)
        batch_scores = self._keras_model([tensor_text, tensor_numerical]).numpy()
        pathogenicity_scores = batch_scores[:, 1]
        if len(pathogenicity_scores) != len(variants):
            raise ValueError(f'Expected same amount of predictions as input data')
        return list(pathogenicity_scores)
