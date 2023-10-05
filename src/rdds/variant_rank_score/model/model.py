# Make sure to set all seeds
from rdds.lib.determinism import enable_determinism; enable_determinism()

from typing import *
import numpy as np
import tensorflow as tf
import os
import datetime
from json import dumps

from .. import WORKDIR
from rdds.lib.hdf5 import Hd5DataGenerator
from rdds.lib.tf import get_tf_dataset_from_hd5_data_generator
from rdds.lib.tf import TextPreprocessingLayer
from rdds.lib.tf import TextVectorizationLayer
from rdds.lib.tf import DnaSequenceTrimmer
from rdds.lib.tf import rejection_resample
from rdds.lib.tf import InstanceNormalisation
from rdds.lib.tf import tfprint

# TODO: Determine whether to use GeneticModels_family_id?
# TODO: Determine whether to use ModelScore_family_id?

# FIXME: IT's required to set this variable to None on first run, to generate a vocabulary. Then restart training using this file.
# See comment in the text_vectorization_layer.py about keras and tensorboard.
_DEFAULT_VOCABULARY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vocabulary.txt'))

FEATURES_TEXT = ['CSQ_PolyPhen',
                 'CSQ_SIFT',
                 'CSQ_CLINVAR_CLNREVSTAT',
                 'CSQ_CLINVAR_CLNSIG',
                 #'FILTER',
                 'most_severe_consequence',
                 'GeneticModels_model'
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
                 embedding_dimensions: int = 10):
        self._features_text: List[str] = features_text
        self._features_numerical = features_numerical
        self._features: List[str] = self._features_text + self._features_numerical
        print(f'Total amount of features: {len(self._features)}')
        self._embedding_dimensions: int = embedding_dimensions
        self._vocabulary_file_path = vocabulary_file_path
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
               dataset: tf.data.Dataset):
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
        preprocessed_dataset = dataset.map(map_func=text_preprocessing_layer)
        input_text_preprocessed = text_preprocessing_layer(input_text)  # -> [bdim, n_words, n_features]
        dna_sequence_trimmer_layer = DnaSequenceTrimmer()
        preprocessed_dataset = preprocessed_dataset.map(map_func=dna_sequence_trimmer_layer)
        input_text_preprocessed = dna_sequence_trimmer_layer(input_text_preprocessed)  # tensor shape preserved

        # Text vectorization
        text_vectorization_layer: TextVectorizationLayer = \
            TextVectorizationLayer(precompiled_vocabulary_file=self._vocabulary_file_path)
        if self._vocabulary_file_path is not None:
            preprocessed_dataset = None  # Don't recompute the vocabulary
        text_vectorization_layer.adapt(dataset=preprocessed_dataset,
                                       embedding_dimensions=self._embedding_dimensions)
        print(f'Vocabulary length: {len(text_vectorization_layer.vocabulary)} words')
        print(text_vectorization_layer.vocabulary)
        text_vectorization_layer.save_vocabulary_to_file(file_path=os.path.join(self._train_log_dir, 'vocabulary.txt'))
        embeddings = text_vectorization_layer(input_text_preprocessed)  # -> [bdim, n_features, n_words, n_embeddings]

        # Automagically convert empty Ragged dimensions to zero-padded Tensors.
        # Required because reduce_prod adds 1.0 to empty Ragged dimensions.
        embeddings = embeddings.to_tensor()

        # Reduce dimensions by computing the average of all words per every feature vector
        embeddings = tf.math.reduce_mean(embeddings, axis=2, keepdims=True)  # -> [bdim, n_features, 1, n_embeddings]

        # Flatten word vector to -> [bdim, n_features * n_embeddings]
        embeddings_flat = tf.reshape(embeddings, (-1, len(self._features_text) * self._embedding_dimensions))

        # Normalization of numerical features (per feature channel)
        # No need to normalize the embeddings since they're nicely distributed
        # Concatenate word vector and numerical features -> [bdim, n_text * n_embeddings + n_numerical]
        complete_feature_vector = tf.keras.layers.Concatenate(axis=1, name='ConcatFeatures')([embeddings_flat,
                                                                                              input_numerical])
        complete_feature_vector = InstanceNormalisation(name='InstanceNormalisation')(complete_feature_vector)
        complete_feature_vector = tf.keras.layers.BatchNormalization()(complete_feature_vector)

        print('Feature vector shape', complete_feature_vector.get_shape())

        # Autoencoder dense layer
        activation = 'relu'
        print('length feature vector', len(self._features))
        units = 512
        delta = int(np.floor(0.1 * units))
        body = tf.keras.layers.Dense(units=units,
                                     activation=activation)(complete_feature_vector)
        body = tf.keras.layers.Dense(units=units - delta,
                                     activation=activation)(body)
        body = tf.keras.layers.Dense(units=units - 2 * delta,
                                     activation=activation)(body)
        body = tf.keras.layers.Dense(units=units - 3 * delta,
                                     activation=activation)(body)
        body = tf.keras.layers.Dense(units=units - 4 * delta,
                                     activation=activation)(body)
        body = tf.keras.layers.Dense(units=units - 5 * delta,
                                     activation=activation)(body)  # -> [bdim, n_units]

        # Specify network out shape
        logits = tf.keras.layers.Dense(units=2, name='Logits', activation='linear')(body)  # -> [bdim, 2]

        # Softmax layer
        confidences = tf.keras.layers.Softmax(name='Confidences')(logits)  # -> [bdim, 2]

        self._keras_model = tf.keras.Model(inputs=[input_text, input_numerical],
                                           outputs=confidences)

        def loss_fn(y_true, y_pred) -> tf.Tensor:
            c = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=False)
            return c

        loss = loss_fn

        metrics = [tf.keras.metrics.TruePositives(),
                   tf.keras.metrics.TrueNegatives(),
                   tf.keras.metrics.FalsePositives(),
                   tf.keras.metrics.FalseNegatives(),
                   tf.keras.metrics.CategoricalAccuracy()]
        optimizer = tf.keras.optimizers.Adam(learning_rate=1E-4)

        def loss_wrapper(x, y, y_pred, sample_weight):
            """
            Wrapper function to inspect loss function.
            """
            y, = y  # unpack tuple
            #x = tfprint(x, 'x')
            #y_pred = tfprint(y_pred, 'y_pred')
            #y = tfprint(y, 'y')
            return self._keras_model.default_loss(x, y, y_pred, sample_weight)

        self._keras_model.default_loss = self._keras_model.compute_loss  # Save loss computation method as default_loss
        self._keras_model.compute_loss = loss_wrapper  # Replace model loss computation with wrapper
        self._keras_model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=metrics)
        self._keras_model.summary(line_length=160)

    def train(self,
              hd5_file_path: str,
              batch_size=128):

        # Set up a training directory for this run
        self._train_log_dir = os.path.join(WORKDIR, 'models/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self._train_log_dir)

        def count_feature_types(hd5_output_dtypes: Dict[str, Type]) -> Tuple[int, int]:
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

        # Training and vocabulary generation setup
        hd5_data_generator_vocabulary: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                           group_name='train',
                                                                           output_tensor_format=[self._features_text])
        hd5_data_generator_train: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                      group_name='train',
                                                                      output_tensor_format=[self._features_text,
                                                                                            self._features_numerical],
                                                                      label='label')
        n_text_features, n_numerical_features = count_feature_types(hd5_output_dtypes=hd5_data_generator_train.data_types)
        input_signature = ((tf.TensorSpec((n_text_features, ), dtype=tf.string),
                           tf.TensorSpec((n_numerical_features, ), dtype=tf.float32, name='input_numerical')),
                           (tf.TensorSpec((2, ), dtype=tf.float32, name='label'), ))
        input_signature_vocabulary = (tf.TensorSpec((n_text_features, ), dtype=tf.string, name='input_text_vocabulary'), )
        dataset_vocabulary: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_vocabulary,
                                                                                     output_signature=input_signature_vocabulary)
        dataset_vocabulary = dataset_vocabulary.prefetch(buffer_size=1024)
        dataset_train: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_train,
                                                                                output_signature=input_signature)
        dataset_train = dataset_train.repeat(-1)
        dataset_train = dataset_train.shuffle(buffer_size=int(hd5_data_generator_train.data_length * 0.01),
                                              seed=1)  # FIXME: Seed
        dataset_train = rejection_resample(dataset=dataset_train,
                                           desired_class_ratio=[0.5, 0.5],
                                           seed=1)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(buffer_size=tf.data.AUTOTUNE)
        # Testing setup
        hd5_data_generator_test: Hd5DataGenerator = Hd5DataGenerator(hd5_file_path=hd5_file_path,
                                                                     group_name='test',
                                                                     output_tensor_format=[self._features_text,
                                                                                           self._features_numerical],
                                                                     label='label')
        dataset_test: tf.data.Dataset = get_tf_dataset_from_hd5_data_generator(hd5_data_generator=hd5_data_generator_test,
                                                                               output_signature=input_signature)
        dataset_test = dataset_test.repeat(-1)
        dataset_test = dataset_test.shuffle(buffer_size=int(hd5_data_generator_test.data_length * 0.01),
                                            seed=1)  # FIXME: Seed
        dataset_test = rejection_resample(dataset=dataset_test,
                                          desired_class_ratio=[0.5, 0.5],
                                          seed=1)
        dataset_test = dataset_test.batch(batch_size)
        dataset_test = dataset_test.prefetch(buffer_size=tf.data.AUTOTUNE)
        print(f'Model Input data mapping: {input_signature}')
        self._build(dataset=dataset_vocabulary)

        # Setup logging and store configuration files
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self._train_log_dir,
                                                              histogram_freq=1,
                                                              embeddings_freq=1)  # FIXME: Bug in Keras
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self._train_log_dir + '/saved-models/{epoch:02d}-{val_loss:.2f}.hdf5',
                                                                 save_freq='epoch',  # At end of epoch, validation metrics are available
                                                                 verbose=1)

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

        history = self._keras_model.fit(x=dataset_train,
                              batch_size=1,
                              epochs=int(1E12),
                              steps_per_epoch=128,
                              validation_data=dataset_test,
                              validation_steps=32,
                              callbacks=[tensorboard_callback, save_model_callback],
                              verbose=2)

    def load_saved_model(self, model_path: str): raise NotImplementedError()

    def predict(self, input_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run predict, inference call on input data dictionary.
        :param input_data: Dictionary of input data. Input shapes, data order
          should conform to earlier established input data format.
        :return: Inferences
        """
        return self._keras_model.predict(x=input_data)
