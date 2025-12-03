from rdds.lib.determinism import enable_determinism

enable_determinism()

# TODO: Refactor below metrics into lib
from rdds.variant_rank_score.model.custom_metrics import MccScore, F1Score
from ..dataset import LABEL_BENIGN, LABEL_PATHOGENIC, DatasetLoader
from .. import WORKDIR
from rdds.lib.logging import get_logger
from rdds.lib.vcf import Variant
from rdds.lib.hpt import HyperParameters
from rdds.lib.tf import AdaptiveLearningRate
from .compute_performance_baseline import compute_performance_baseline, \
    StaticRecall, StaticPrecision, StaticTruePositives, StaticTrueNegatives, \
    StaticFalsePositives, StaticFalseNegatives, StaticAUC, StaticF1Metric, StaticMccMetric
from .default_model import DEFAULT_MODEL_SPEC

_LOGGER = get_logger(name="GICAM", log_level="info")

from typing import Dict, List, Union
import numpy as np
import tensorflow as tf
import os
import datetime
import matplotlib.pyplot as plt


def assert_nonneg_norm(x: tf.Tensor):
    """
    Check input tensor for all values in range [0, 1]
    """
    assert_nonneg_op = tf.debugging.assert_non_negative(x)
    assert_max_op = tf.debugging.assert_less_equal(x, 1.0)
    with tf.control_dependencies([assert_nonneg_op, assert_max_op]):
        return x


class GenmodCorrectionLayer(tf.keras.layers.Dense):
    """
    Layer to adjust Genmod inferences with a scaler and bias for optimal performance.

    Use sigmoid fn to make sure output is in [0, 1].
    """

    def __init__(self, *args, **kwargs):
        kwargs.update({"units": 1, "use_bias": True, "activation": "sigmoid"})
        super().__init__(*args, **kwargs)

    def call(self, score):
        y = super().call(score)
        y = assert_nonneg_norm(y)
        return y


class NonNegNormConstraint(tf.keras.constraints.Constraint):
    """
    Constraint w to range [0, 1]
    """

    def __call__(self, w):
        nonneg = tf.keras.constraints.NonNeg()
        w = nonneg(w)
        norm = tf.keras.constraints.MinMaxNorm(0, 1)
        w = norm(w)
        return w

class NonNegativeConstraint(tf.keras.constraints.Constraint):

    def __call__(self, w):
        return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)


class NonPositiveConstraint(tf.keras.constraints.Constraint):

    def __call__(self, w):
        return w * tf.cast(tf.math.less(w, 0.), w.dtype)


class HarmonicMeanLayer(tf.keras.layers.Layer):
    """
    Layer to compute a joint, potentially biased combination score from two input scores [0, 1].
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.harmonic_scaler = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(1.0),
            constraint=NonNegNormConstraint(),
            dtype=tf.float32,
            trainable=True,
            name="harmonic_mean_scaler",
        )

    def call(self, a, b):
        numerator = (1 + self.harmonic_scaler**2) * a * b
        denominator = ((self.harmonic_scaler**2) * a) + b
        harmonic_mean = tf.math.divide_no_nan(numerator, denominator, name="harmonic_mean")
        harmonic_mean = assert_nonneg_norm(harmonic_mean)
        return harmonic_mean

class Boundary(tf.keras.layers.Layer):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)
        self.b_genmod = self.add_weight(
            name='b_genmod',
            shape=(1, ),
            initializer=tf.keras.initializers.random_normal(),
            #constraint=NonPositiveConstraint(),
            dtype=tf.float32,
            trainable=True
        )
        self.w_genmod = self.add_weight(
            name='w_genmod',
            shape=(1, ),
            initializer=tf.keras.initializers.random_normal(),
            #constraint=NonNegativeConstraint(),
            dtype=tf.float32,
            trainable=True
        )
        self.b_mivmir = self.add_weight(
            name='b_mivmir',
            shape=(1, ),
            initializer=tf.keras.initializers.random_normal(),
            #constraint=NonPositiveConstraint(),
            dtype=tf.float32,
            trainable=True
        )
        self.w_mivmir = self.add_weight(
            name='w_mivmir',
            shape=(1, ),
            initializer=tf.keras.initializers.random_normal(),
            #constraint=NonNegativeConstraint(),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, mivmir, genmod, training=False):
        # Add decision boundary for MIVMIR
        mivmir_filtered = self.w_mivmir * mivmir + self.b_mivmir

        # Add decision boundary for Genmod
        genmod_filtered = self.w_genmod * genmod + self.b_genmod

        # Cap into [0, 1]
        genmod_discreet = tf.keras.activations.relu(genmod_filtered,
                                                    max_value=1.0,
                                                    threshold=0.0)
        mivmir_discreet = tf.keras.activations.relu(mivmir_filtered,
                                                    max_value=1.0,
                                                    threshold=0.0)

        # Is now a "capped" [0, 1] map of good regions in [mivmir, genmod] coordinates for reducing FPR
        transfer_fn = mivmir_discreet * genmod_discreet
        return transfer_fn


class ThresholdedScore(tf.keras.layers.Layer):

    """
    A layer to optimize Genmod input decision boundary.

    There's a threshold that's optimal for removing FPs.
    """

    def __init__(self,
                 initial_b_genmod,
                 initial_w_genmod,
                 initial_b_mivmir,
                 initial_w_mivmir,
                 *args,
                 **kwargs):
        super().__init__(*args,
                         **kwargs)
        self.n = 1
        self.b_genmod = self.add_weight(
            name='b_genmod',
            shape=(self.n, ),
            #initializer=tf.keras.initializers.random_normal(mean=-3.4),
            initializer=tf.keras.initializers.Constant(initial_b_genmod),
            dtype=tf.float32,
            trainable=True
        )
        self.w_genmod = self.add_weight(
            name='w_genmod',
            shape=(self.n, ),
            #initializer=tf.keras.initializers.random_normal(mean=-3.2),
            initializer=tf.keras.initializers.Constant(initial_w_genmod),
            dtype=tf.float32,
            trainable=True
        )
        self.b_mivmir = self.add_weight(
            name='b_mivmir',
            shape=(self.n, ),
            #initializer=tf.keras.initializers.random_normal(mean=-3.8),
            initializer=tf.keras.initializers.Constant(initial_b_mivmir),
            dtype=tf.float32,
            trainable=True
        )
        self.w_mivmir = self.add_weight(
            name='w_mivmir',
            shape=(self.n, ),
            #initializer=tf.keras.initializers.random_normal(mean=-3.4),
            initializer=tf.keras.initializers.Constant(initial_w_mivmir),
            dtype=tf.float32,
            trainable=True
        )

    def call(self, mivmir, genmod, training=False):
        #breakpoint()
        mivmir_boundaries = self.w_mivmir * mivmir + self.b_mivmir
        genmod_boundaries = self.w_genmod * genmod + self.b_genmod
        # Cap into [0, 1]
        if training:
            alpha=0.01  # Required for converging
        else:
            alpha=0  # Don't allow negative values in output value
        cap_genmod = tf.keras.activations.relu(genmod_boundaries,
                                               max_value=1.0,
                                               threshold=0.0,
                                               alpha=alpha)
        cap_mivmir = tf.keras.activations.relu(mivmir_boundaries,
                                               max_value=1.0,
                                               threshold=0.0,
                                               alpha=alpha)

        reduced_genmod = tf.reduce_sum(cap_genmod, axis=-1, keepdims=True)
        reduced_mivmir = tf.reduce_sum(cap_mivmir, axis=-1, keepdims=True)

        transfer_fn = reduced_genmod * cap_mivmir
        transfer_fn = reduced_mivmir * reduced_genmod
        # Transpose one of vactors to produce 1D value as means of reduction
        #breakpoint()
        #if self.n > 1:
        #    transfer_fn = tf.reduce_sum(transfer_fn, axis=-1, keepdims=True)

        if training:
            return transfer_fn
        else:
            # During inference convolve mivmir score with transfer function
            return mivmir * transfer_fn


class Gicam:
    """
    Model to optimally join GENMOD and MIVMIR inferences.
    """

    def __init__(self,
                 work_dir: str = WORKDIR,
                 train_max_epochs: int = 220):
        self._train_log_dir = os.path.join(
            work_dir, "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(self._train_log_dir, exist_ok=True)
        self._train_max_epochs = train_max_epochs

    @property
    def input_spec(self):
        # return x, y where x i training data spec and y is label name
        return ("score_mivmir", "score_genmod"), ("pathogenic",)

    def _build_model(self, hparams: HyperParameters):
        score_mivmir = tf.keras.Input(shape=(1,), dtype=tf.float32, name="score_mivmir")
        score_genmod = tf.keras.Input(shape=(1,), dtype=tf.float32, name="score_genmod")

        initial_b_genmod = hparams.Float('initial_b_genmod',
                                  min_value=-30,
                                  max_value=0,
                                  default=-0.65)
        initial_w_genmod = hparams.Float('initial_w_genmod',
                                  min_value=0,
                                  max_value=30,
                                  default=2.35)
        initial_b_mivmir = hparams.Float('initial_b_mivmir',
                                  min_value=-30,
                                  max_value=0,
                                  default=0)
        initial_w_mivmir = hparams.Float('initial_w_mivmir',
                                  min_value=0,
                                  max_value=30,
                                  default=2.7)
        threshold_layer = ThresholdedScore(initial_b_genmod=initial_b_genmod,
                                           initial_w_genmod=initial_w_genmod,
                                           initial_b_mivmir=initial_b_mivmir,
                                           initial_w_mivmir=initial_w_mivmir)
        y = threshold_layer(mivmir=score_mivmir, genmod=score_genmod)

        model = tf.keras.Model(
            inputs=[score_mivmir, score_genmod], outputs=[y]
        )

        metrics = [
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            MccScore(),
            F1Score()
        ]
        metrics.extend(self._baseline_metrics)

        adaptive_network_param = hparams.Fixed('adaptive-LR-network-param',
                                               value=0)
        if adaptive_network_param > 0:
            writer = tf.summary.create_file_writer(os.path.join(self._train_log_dir, 'metrics'))
            self._adaptive_learning_rate_cb = AdaptiveLearningRate(network_param=adaptive_network_param,
                                                                   warmup_epochs=1,
                                                                   writer=writer)
        else:
            self._adaptive_learning_rate_cb = None

        learning_rate = hparams.Float('learning-rate',
                                       min_value = 1E-7,
                                       max_value = 1E-3,
                                       default = 1E-2,
                                       step = 10,
                                       sampling = 'log')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),  # Possibly also set by AdaptiveLearningRate
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=metrics,
        )
        model.summary(line_length=160)

        self._keras_model = model

    def save_model_fn(self, epoch, logs: dict):
        """
        Saves model to Keras saved model format
        :param epoch: Current epoch
        :param logs: Dictionary of batch statistics
        """
        # NOTE: The suffix .keras is important to tf.keras.saving.load_model()
        filepath = self._train_log_dir + "/saved-models/%d-%.4f.keras" % (
            epoch,
            logs.get('val_loss', np.nan),
        )
        _LOGGER.info(f"Saving model to {filepath}")
        self._keras_model.save(filepath=filepath)

    @staticmethod
    def from_saved_model(
        model_path: str = DEFAULT_MODEL_SPEC.keras_model,
    ):
        """
        Load trained model from model_path
        :param model_path: Path to Keras saved model (*.keras) zip file
        """
        # Load Keras Model
        _LOGGER.info(f"Loading model: {DEFAULT_MODEL_SPEC}")
        with tf.keras.saving.custom_object_scope(
            {
                "MccScore": MccScore,
                "F1Score": F1Score,
                "GenmodCorrectionLayer": GenmodCorrectionLayer,
                "HarmonicMeanLayer": HarmonicMeanLayer,
                "ThresholdedScore": ThresholdedScore,
                "StaticPrecision": StaticPrecision,
                "StaticRecall": StaticRecall,
                "StaticF1Metric": StaticF1Metric,
                "StaticMccMetric": StaticMccMetric,
                "StaticAUC": StaticAUC,
                "StaticTruePositives": StaticTruePositives,
                "StaticTrueNegatives": StaticTrueNegatives,
                "StaticFalseNegatives": StaticFalseNegatives,
                "StaticFalsePositives": StaticFalsePositives
            }
        ):
            keras_model = tf.keras.saving.load_model(model_path)
        _LOGGER.info(f"Model input: {keras_model.inputs}")
        _LOGGER.info(f"Model output: {keras_model.outputs}")
        keras_model.summary(line_length=160)
        model = Gicam()
        model._keras_model = keras_model
        return model

    def _build_dataset(self,
                       path_to_hd5_dataset: str,
                       hparams: HyperParameters,
                       amount_data: float = 1.0) -> (tf.data.Dataset, tf.data.Dataset):
        """

        :param path_to_hd5_dataset: Path to hd5 file containing training data
        :param hparams:
        :param amount_data: Ratio (0, 1] of data to use (good for testing is 0.025)
        """
        dataset_loader = DatasetLoader(path_to_hd5_dataset=path_to_hd5_dataset, amount_data=amount_data)
        _LOGGER.info(f"Dataset load configs: {dataset_loader}")
        x_train, y_train = dataset_loader.get_train_data(self.input_spec)
        x_test, y_test = dataset_loader.get_test_data(self.input_spec)

        # Compute baseline metrics
        # NOTE: These metric are on the complete, unmodified dataset
        # Sampling etc do change the constitution of the training, test datasets and this will be reflected in metrics.
        baseline_metrics = compute_performance_baseline(dataset_loader=dataset_loader)

        def assemble_data_from_input_tensors(
                x: np.ndarray, tensor_names: List[str]
        ) -> Dict[str, np.ndarray]:
            d = dict()
            for col_idx, name in enumerate(tensor_names):
                d.update({name: x[:, col_idx:col_idx + 1]})
            return d

        input_tensors, _ = self.input_spec
        x = assemble_data_from_input_tensors(x=x_train, tensor_names=input_tensors)
        y = y_train[:, 0]

        x_val = assemble_data_from_input_tensors(x=x_test, tensor_names=input_tensors)
        y_val = y_test[:, 0]

        batch_size = hparams.Choice('batch-size',
                                    values=[2**7, 2**8, 2**9, 2**10, 2**11, 2**12, 2**13, 2**14],
                                    default=2**12)

        dataset_train = tf.data.Dataset.from_tensor_slices((x, y))

        # Downsample TNs in training dataset
        sample_pathogenic_with_likelihood = hparams.Float('sample_pathogenic_with_likelihood',
                                                          min_value=0,
                                                          max_value=0.9,
                                                          default=0.5)

        """
        Setup TN weights to preserve TN downsampling bias.

        Additionally, given Genmod behavior (genmod is suitable for FPR reduction only),
        tune tp_weight so that recall is kept at 1.0 or close to (mivmir is not ideal classifier, expect some FNs).
        Given max(recall), find best precision. Increasing tp_weight improves recall.
        """
        tp_weight = 100 * (0.5 / sample_pathogenic_with_likelihood)
        tn_weight = 1 * (sample_pathogenic_with_likelihood / 0.5)
        _LOGGER.info(f"TN weight is {tn_weight}")
        _LOGGER.info(f"TP weight is {tp_weight}")
        _LOGGER.info(f"nTPs: {dataset_loader.amount_train_pathogenic_samples}")
        @tf.function
        def _add_weight(data, labels, **kwargs):
            is_pathogenic = tf.equal(labels, LABEL_PATHOGENIC)
            weights = tf.where(condition=is_pathogenic,
                               x=tf.ones_like(labels) * tf.constant(tp_weight, dtype=tf.float32),  # cond == True
                               y=tf.ones_like(labels, dtype=tf.float32) * tf.constant(tn_weight, dtype=tf.float32))  # cond == False
            return data, labels, weights
        dataset_train = dataset_train.map(map_func=lambda *args: _add_weight(*args),
                                          num_parallel_calls=tf.data.AUTOTUNE)
        dataset_train = dataset_train.cache()

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
            predicate = tf.equal(labels, target_label)
            return predicate

        _LOGGER.info(
            f'Sampling pathogenic variants during training with likelihood of {sample_pathogenic_with_likelihood}')
        sampling_weights = (1.0 - sample_pathogenic_with_likelihood, sample_pathogenic_with_likelihood)
        _LOGGER.info(f'Sampling weights (benign, pathogenic): {sampling_weights}')
        train_pathogenic_variants = \
            dataset_train.filter(predicate=lambda *args: filt_fn(*args, target_label=LABEL_PATHOGENIC))
        train_benign_variants = \
            dataset_train.filter(predicate=lambda *args: filt_fn(*args, target_label=LABEL_BENIGN))
        train_pathogenic_variants = train_pathogenic_variants.cache()
        train_benign_variants = train_benign_variants.cache()
        train_pathogenic_variants=train_pathogenic_variants.shuffle(dataset_loader.amount_train_pathogenic_samples, seed=1)
        train_benign_variants=train_benign_variants.shuffle(dataset_loader.dlen_train - dataset_loader.amount_train_pathogenic_samples, seed=1)
        dataset_train = tf.data.Dataset.sample_from_datasets(
            datasets=(train_benign_variants, train_pathogenic_variants),
            weights=sampling_weights,
            stop_on_empty_dataset=True,  ### IMPORTANT ###
            seed=1)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(tf.data.AUTOTUNE)

        dataset_validation = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        dataset_validation = dataset_validation.cache()
        if amount_data < 1:
            # Shuffle the data to provide representative minibatches
            dataset_validation = dataset_validation.shuffle(buffer_size=dataset_loader.dlen_test, seed=1)
        dataset_validation = dataset_validation.batch(batch_size)
        dataset_validation = dataset_validation.prefetch(tf.data.AUTOTUNE)

        self._dataset_train = dataset_train
        self._dataset_validation = dataset_validation
        self._baseline_metrics = baseline_metrics

    def build(self,
              path_to_hd5_dataset: str,
              hparams: HyperParameters,
              amount_data=0.1):
        self._build_dataset(path_to_hd5_dataset=path_to_hd5_dataset,
                            hparams=hparams,
                            amount_data=amount_data)
        self._build_model(hparams=hparams)

    def train(self,
              hparam_tuning_callbacks: List[tf.keras.callbacks.Callback] = None,
              validation_only_beginning_end=False):
        """
        Train model
        :param hparam_tuning_callbacks: Callbacks for hyperparameter tuner
        :param validation_only_beginning_end: Run validation only on train start and end
        """

        callbacks: List[tf.keras.callbacks.Callback] = list()
        # Setup default monitoring in Tensorboard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self._train_log_dir, histogram_freq=1, embeddings_freq=0
            )
        )
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(on_epoch_end=self.save_model_fn)
        )
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        #if not validation_only_beginning_end:
            #callbacks.append(tf.keras.callbacks.EarlyStopping(start_from_epoch=2))
        if hparam_tuning_callbacks:
            callbacks.extend(hparam_tuning_callbacks)

        if self._adaptive_learning_rate_cb:
            callbacks.append(self._adaptive_learning_rate_cb)

        validation_freq = 1
        if validation_only_beginning_end:
            validation_freq = [1, self._train_max_epochs]

        history = self._keras_model.fit(
            self._dataset_train,
            validation_data=self._dataset_validation,
            epochs=self._train_max_epochs,
            validation_freq=validation_freq,
            callbacks=callbacks,
            verbose=2,
        )

        print("Final weights:")
        for layer in self._keras_model.layers:
            print(f"{layer}")
            print(f"{layer.weights}")

        return history

    def score_variants(self, variants: List[Variant]) -> np.ndarray:
        """
        Score a batch of variants.
        :param variants: List of Variants
        :return: Scores as 1D array (sorted according to ordering in variants)
        """
        n_samples = len(variants)
        score_mivmir: np.ndarray = np.zeros((n_samples, 1))  # [batch_dim, feature_dim]
        score_genmod: np.ndarray = np.zeros((n_samples, 1))
        for i, variant in enumerate(variants):
            score_mivmir[i, 0] = variant.INFO['VrsModelPrediction']
            rank_score_normalized_str: str = variant.INFO['RankScoreNormalized']  # format: str: case_name:rank_score
            rank_score = float(rank_score_normalized_str.split(':')[1])
            score_genmod[i, 0] = rank_score
        scores = self._keras_model.predict_on_batch({'score_genmod': score_genmod,
                                                     'score_mivmir': score_mivmir})
        scores = scores[:, 0]  # Return scores as 1D array (no batch dimension)
        return scores

    def visualize_decision_boundary(self,
                                    storage_path: str = None,
                                    show=False):
        """
        Visualize model decision boundary on 2D plot, mivmir score vs genmod score
        :param storage_path: 'train-log-dir' or directory path
        """
        scores = np.linspace(start=0, stop=1, num=30)
        x = []
        y = []
        z = []
        z_train = []
        clr = []
        clr_train = []
        for score_mivmir in scores:
            for score_genmod in scores:
                score = self._keras_model({'score_mivmir': np.array([score_mivmir]),
                                                   'score_genmod': np.array([score_genmod])}).numpy()[0, 0]
                score_train = self._keras_model({'score_mivmir': np.array([score_mivmir]),
                                                   'score_genmod': np.array([score_genmod])}, training=True).numpy()[0, 0]
                x.append(score_mivmir)
                y.append(score_genmod)
                z.append(score)
                z_train.append(score_train)
                color = (0.1, 0.1, 0.1, 0.25)  # Grey
                color_train = (0.1, 0.1, 0.1, 0.25)  # Grey
                if score > 0.5:
                    color = (1.0, 0, 0 , 1.0)  # Red
                clr.append(color)
                if score_train > 0.5:
                    color_train = (1.0, 0, 0 , 1.0)  # Red
                clr_train.append(color_train)

        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(2, 2, 1)
        ax.scatter(x, y, c=clr)
        ax.set_xlabel('Score Mivmir')
        ax.set_ylabel('Score Genmod')
        ax.legend(['Grey (benign), Red (Pathogenic)'])
        ax = fig.add_subplot(2, 2, 2)
        ax.scatter(x, y, c=clr_train)
        ax.set_xlabel('Score Mivmir')
        ax.set_ylabel('Score Genmod')
        ax.legend(['Grey (benign), Red (Pathogenic) (train)'])
        fig.suptitle('Decision Boundary')
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(x, y, z, c=clr)
        ax.legend(['Grey (benign), Red (Pathogenic)'])
        ax.set_xlabel('Score Mivmir')
        ax.set_ylabel('Score Genmod')
        ax.set_zlabel('Score Gicam')
        ax.set_zlim(-0.1, 1.1)
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(x, y, z_train, c=clr_train)
        ax.legend(['Grey (benign), Red (Pathogenic) (train)'])
        ax.set_xlabel('Score Mivmir')
        ax.set_ylabel('Score Genmod')
        ax.set_zlabel('Score Gicam')
        ax.set_zlim(-0.1, 1.1)
        if storage_path:
            if storage_path == 'train-log-dir':
                fig_path = os.path.join(self._train_log_dir, 'decision-boundary.png')
            else:
                fig_path = os.path.join(storage_path, 'decision-boundary.png')
            fig.savefig(fig_path)
        if show:
            plt.show()