from rdds.lib.determinism import enable_determinism

enable_determinism()
# TODO: Refactor below metrics into lib
from rdds.variant_rank_score.model.custom_metrics import MccScore, F1Score
from ..dataset import LABEL_BENIGN, LABEL_PATHOGENIC, DatasetLoader
from .. import WORKDIR
from rdds.lib.logging import get_logger

_LOGGER = get_logger(name="GICAM", log_level="info")

from typing import Dict, List
import numpy as np
import tensorflow as tf
import os
import datetime

# TODO: Explore performance of softmax function (compare to harmonic mean)


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
        harmonic_mean = tf.math.divide(numerator, denominator, name="harmonic_mean")
        harmonic_mean = assert_nonneg_norm(harmonic_mean)
        return harmonic_mean


class Gicam:
    """
    Model to optimally join GENMOD and MIVMIR inferences.
    """

    def __init__(self):
        self._train_log_dir = os.path.join(
            WORKDIR, "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(self._train_log_dir, exist_ok=True)

    @property
    def input_spec(self):
        # return x, y where x i training data spec and y is label name
        return ("score_mivmir", "score_genmod"), ("pathogenic",)

    def _build(self):
        score_mivmir = tf.keras.Input(shape=(1,), dtype=tf.float32, name="score_mivmir")
        score_genmod = tf.keras.Input(shape=(1,), dtype=tf.float32, name="score_genmod")

        genmod_correction_layer = GenmodCorrectionLayer()
        genmod_corrected = genmod_correction_layer(score_genmod)

        harmonic_mean_layer = HarmonicMeanLayer()
        harmonic_mean = harmonic_mean_layer(a=genmod_corrected, b=score_mivmir)

        model = tf.keras.Model(
            inputs=[score_mivmir, score_genmod], outputs=[harmonic_mean]
        )

        metrics = [
            tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives(),
            tf.keras.metrics.FalsePositives(),
            tf.keras.metrics.FalseNegatives(),
            MccScore(),
            F1Score(),
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-2),
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
            logs["val_loss"],
        )
        _LOGGER.info(f"Saving model to {filepath}")
        self._keras_model.save(filepath=filepath)

    @staticmethod
    def from_saved_model(
        model_path: str,
    ):
        """
        Load trained model from model_path
        :param model_path: Path to Keras saved model (*.keras) zip file
        """
        # Load Keras Model
        with tf.keras.saving.custom_object_scope(
            {
                "MccScore": MccScore,
                "F1Score": F1Score,
                "GenmodCorrectionLayer": GenmodCorrectionLayer,
                "HarmonicMeanLayer": HarmonicMeanLayer,
            }
        ):
            model = tf.keras.saving.load_model(model_path)
        _LOGGER.info(f"Model input: {model.inputs}")
        _LOGGER.info(f"Model output: {model.outputs}")
        model.summary(line_length=160)
        model = Gicam()
        model._keras_model = model
        return model

    def train(self, path_to_dataset: str):
        dataset_loader = DatasetLoader(path_to_dataset=path_to_dataset)

        x_train, y_train = dataset_loader.get_train_data(self.input_spec)
        x_test, y_test = dataset_loader.get_test_data(self.input_spec)

        def assemble_data_from_input_tensors(
            x: np.ndarray, tensor_names: List[str]
        ) -> Dict[str, np.ndarray]:
            d = dict()
            for col_idx, name in enumerate(tensor_names):
                d.update({name: x[:, col_idx]})
            return d

        input_tensors, _ = self.input_spec
        x = assemble_data_from_input_tensors(x=x_train, tensor_names=input_tensors)
        y = y_train[:, 0]

        x_val = assemble_data_from_input_tensors(x=x_test, tensor_names=input_tensors)
        y_val = y_test[:, 0]

        self._build()

        callbacks: List[tf.keras.callbacks.Callback] = list()
        # Setup default monitoring in Tensorboard
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self._train_log_dir, histogram_freq=1, embeddings_freq=1
            )
        )
        callbacks.append(
            tf.keras.callbacks.LambdaCallback(on_epoch_end=self.save_model_fn)
        )
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
        callbacks.append(tf.keras.callbacks.EarlyStopping())

        self._keras_model.fit(
            x=x,
            y=y,
            validation_data=(x_val, y_val),
            batch_size=1,
            epochs=200,
            callbacks=callbacks,
            verbose=2,
        )

        print("Final weights:")
        for layer in self._keras_model.layers:
            print(f"{layer}")
            print(f"{layer.weights}")
