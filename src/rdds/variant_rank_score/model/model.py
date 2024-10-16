# Make sure to set all seeds
from rdds.lib.determinism import enable_determinism; enable_determinism()

from typing import *
import numpy as np
import tensorflow as tf
import os
import datetime
from json import dumps
import logging

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

tf.debugging.enable_check_numerics()  # Raises exception on +/- INF and NaNs in tensors

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
                 vocabulary_file_path: str = _DEFAULT_VOCABULARY_FILE,
                 numerical_normalisation_weights_file_path: str = _DEFAULT_NUMERICAL_NORMALISATION_WEIGHTS,
                 embedding_dimensions: int = 10):
        self._features_text: List[str] = features_text
        self._features_numerical = features_numerical
        self._features: List[str] = self._features_text + self._features_numerical
        _LOGGER.info(f'Total amount of features: {len(self._features)}')
        self._embedding_dimensions: int = embedding_dimensions
        self._vocabulary_file_path = vocabulary_file_path
        self._numerical_normalisation_weights_file_path = numerical_normalisation_weights_file_path
        self._keras_model: tf.keras.Model = None
        self._train_log_dir: str = None

    @staticmethod
    def preprocess_filter_textual_features(*tensors: Tuple[tf.RaggedTensor]) -> Tuple[tf.RaggedTensor]:
        """
        Select Tensors with dtype tf.string in dataset for further downstream processing.
        :param tensors: Tuple of RaggedTensor
        :return: Tuple of RaggedTensors, only tf.string dtype tensors
        """
        text_feature_tensors = tuple()
        for tensor in tensors:
            is_text_feature: bool = True if tensor.dtype == tf.string else False
            if is_text_feature:
                text_feature_tensors += (tensor, )
        return text_feature_tensors

    def _build(self,
               text_dataset: tf.data.Dataset = None,
               numerical_dataset: tf.data.Dataset = None):
        """
        :param text_dataset: Optional datset to use for compiling vocabulary
        :param numerical_dataset: Optional dataset to compile normalisation factors
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
            text_dataset.map(map_func=text_preprocessing_layer) if text_dataset else None
        input_text_preprocessed = text_preprocessing_layer(input_text)  # -> [bdim, n_words, n_features]
        dna_sequence_trimmer_layer = DnaSequenceTrimmer()
        preprocessed_dataset = \
            preprocessed_dataset.map(map_func=dna_sequence_trimmer_layer) if preprocessed_dataset else None
        input_text_preprocessed = dna_sequence_trimmer_layer(input_text_preprocessed)  # tensor shape preserved

        feature_selection_regularizer = tf.keras.regularizers.L1(0.00001)  # L1 regularizer to perform feature selection

        # Text vectorization
        precompiled_vocabulary_file = None if preprocessed_dataset else self._vocabulary_file_path
        embeddings_layer: EmbeddingsReductionLayer = \
            EmbeddingsReductionLayer(precompiled_vocabulary_file=precompiled_vocabulary_file,
                                     embedding_dimensions=self._embedding_dimensions,
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
        if numerical_dataset:
            _LOGGER.info('Adapting normalisation layer from dataset')
            numerical_normalisation_layer.adapt_from_dataset(data=numerical_dataset)
        else:
            _LOGGER.info(f'Loading normalisation weights from file {self._numerical_normalisation_weights_file_path}')
            numerical_normalisation_layer.load_saved_weights_file(file_path=self._numerical_normalisation_weights_file_path)
        # Store normalisation weights to training output dir
        normalisation_weights_file_path: str = os.path.join(self._train_log_dir, 'normalisation.tar')
        numerical_normalisation_layer.save_weights_to_file(file_path=normalisation_weights_file_path)
        _LOGGER.info(f'Saved normalisation weights to {normalisation_weights_file_path}')

        # Flatten word vector to -> [bdim, n_features * n_embeddings]
        embeddings_flat = tf.reshape(embeddings, (-1, len(self._features_text) * self._embedding_dimensions))

        # Normalization of numerical features (per feature channel)
        # No need to normalize the embeddings since they're nicely distributed
        # Concatenate word vector and numerical features -> [bdim, n_text * n_embeddings + n_numerical]
        regularizer = tf.keras.regularizers.L2(0.000000001)  # L2; regularisation to deal with multicollinearity
        embeddings_branch = tf.keras.layers.Dense(units=128, activation='relu', kernel_regularizer=None)(embeddings_flat)
        embeddings_branch = tf.keras.layers.Dense(units=84, activation='relu', kernel_regularizer=None)(embeddings_branch)
        numerical_branch = tf.keras.layers.Dense(units=128,activation='relu',kernel_regularizer=feature_selection_regularizer)(input_numerical_normalized)
        numerical_branch = tf.keras.layers.Dense(units=84, activation='relu', kernel_regularizer=None)(numerical_branch)
        complete_feature_vector = tf.keras.layers.Concatenate(axis=1, name='ConcatFeatures')([embeddings_branch,
                                                                                              numerical_branch])
        _LOGGER.info(f'Feature vector shape {complete_feature_vector.get_shape()}')

        # Autoencoder dense layer
        activation = 'relu'
        _LOGGER.info(f'length feature vector {len(self._features)}')
        units = 512
        delta = int(np.floor(0.1 * units))
        body = tf.keras.layers.Dense(units=units,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(complete_feature_vector)
        body = tf.keras.layers.Dense(units=units - delta,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(body)
        body = tf.keras.layers.Dense(units=units - 2 * delta,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(body)
        body = tf.keras.layers.Dense(units=units - 3 * delta,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(body)
        body = tf.keras.layers.Dense(units=units - 4 * delta,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(body)
        body = tf.keras.layers.Dense(units=units - 5 * delta,
                                     activation=activation,
                                     kernel_regularizer=regularizer)(body)  # -> [bdim, n_units]

        # Specify network out shape
        logits = tf.keras.layers.Dense(units=2, name='Logits', activation='linear')(body)  # -> [bdim, 2]

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
        optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4)

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

    def train(self,
              hd5_file_path: str,
              compile_vocabulary_normalisation_factors: bool = True,
              batch_size=128):

        # Set up a training directory for this run
        self._train_log_dir = os.path.join(WORKDIR, 'models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self._train_log_dir)

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
        self._build(text_dataset=dataset_vocabulary,
                    numerical_dataset=dataset_numerical)

        steps_per_epoch = int(np.ceil(float(hd5_data_generator_train.data_length) / float(batch_size)))

        # Setup logging and store configuration files
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self._train_log_dir,
                                                              histogram_freq=1,
                                                              embeddings_freq=1)  # FIXME: Bug in Keras
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self._train_log_dir + '/saved-models/{epoch:02d}-{val_loss:.2f}.hdf5',
                                                                 save_freq='epoch',  # At end of epoch, validation metrics are available
                                                                 verbose=1)

        def save_model_fn(epoch: int, logs=Optional[dict]):
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

        save_model_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=save_model_fn)

        # TODO: A keras.save_model callback
        compile_config: Dict[str, Any] = self._keras_model.get_compile_config()
        network_config: str = self._keras_model.to_json()
        with open(os.path.join(self._train_log_dir, 'compile-config.txt'), 'w') as file:
            file.write(dumps(compile_config))
        with open(os.path.join(self._train_log_dir, 'network-config.txt'), 'w') as file:
            file.write(network_config)
        with open(os.path.join(self._train_log_dir, 'dataset-config.txt'), 'w') as file:
            file.write('Dataset train:\n' + str(vars(dataset_train)))
            file.write('Dataset test:\n' + str(vars(dataset_test)))
        with open(os.path.join(self._train_log_dir, 'build-config.txt'), 'w') as file:
            file.write(str(globals()))
            file.write(str(locals()))

        validation_steps = int(np.ceil(float(hd5_data_generator_test.data_length) / float(batch_size)))

        history = self._keras_model.fit(x=dataset_train,
                              batch_size=1,
                              epochs=int(1E12),
                              steps_per_epoch=steps_per_epoch,
                              validation_data=dataset_test,
                              validation_steps=validation_steps,
                              callbacks=[tensorboard_callback,
                                         save_model_callback,
                                         tf.keras.callbacks.TerminateOnNaN()],
                              verbose=2)

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
                       group_name: str = 'train'):

        np.set_printoptions(linewidth=128, precision=6, floatmode='fixed')

        # TODO: Make sure config to Hd5DataGenerator is identical to train time setup
        datagen: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                     group_name=group_name,
                                                     output_tensor_format=[self._features_text,
                                                                           self._features_numerical],
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

        output_file = open('/rdds/preds.tsv', 'w')  # FIXME: path
        output_file.write('Data features: ' + str(datagen._output_tensor_format)+'\n')
        output_file.write('feature_text\tfeature_numerical\ttruth\tpredicted\n')

        nr_samples = float(datagen.data_length)
        processed_samples: float = 0.0
        for data, labels in dataset.as_numpy_iterator():
            label, = labels
            tensor_str, tensor_numerical = data
            r = self._keras_model([tensor_str, tensor_numerical])
            batch_size = tensor_str.shape[0]
            for batch_idx in range(0, batch_size):
                formatted = ''
                formatted += f'{tensor_str[batch_idx]}\t'.replace('\n', '')
                formatted += f'{tensor_numerical[batch_idx]}\t'.replace('\n', '')
                formatted += f'{label[batch_idx, 1]}\t'
                formatted += f'{r.numpy()[batch_idx, 1]}\n'
                output_file.write(formatted)
                processed_samples += 1
            _LOGGER.info("%.2f" % (100.0 * (processed_samples / nr_samples)))
            output_file.flush()
        output_file.close()
