import tensorflow as tf
import keras_tuner
from keras_tuner.src import backend as keras_tuner_backend
from keras_tuner import HyperParameters
from tensorboard.plugins.hparams import api as hparams_api
from keras_tuner.src.engine import tuner_utils
from typing import Callable

from rdds.lib.logging import get_logger
_LOGGER = get_logger('keras_tuner', 'info')


class CustomTuner(keras_tuner.Tuner):
    """
    A custom tuner instance that wraps model creation and evaluation step.

    Generates hyperparameter evaluation runs like:
    /tmp/tmp8ivqquvg
        `-- GridSearchTuner
            |-- oracle.json
            |-- trial_0
            |   |-- events.out.tfevents.1729517596
            |   |-- train
            |   |   `-- events.out.tfevents.1729517596
            |   |-- trial.json
            |   `-- validation
            |       `-- events.out.tfevents.1729517599.
            |-- trial_1
            |   |-- events.out.tfevents.1729517600
            |   |-- train
            |   |   `-- events.out.tfevents.1729517600
            |   |-- trial.json
            |   `-- validation
            |       `-- events.out.tfevents.1729517603
            `-- tuner0.json
    """

    def __init__(self,
                 build_fn: Callable,
                 fit_fn: Callable,
                 log_dir: str,
                 *args,
                 objective_metric: str = 'val_loss',
                 **kwargs):
        """
        :param build_fn: A function that has the signature build_fn(hparams: keras_tuner.HyperParameters)
          that returns a compiled instance of keras.model.Model
        :param fit_fn: A function that runs model training and evaluation on N epochs, with signature
          fit(model: tf.keras.models.Model, fit_callbacks: List[tf.keras.callbacks.Callback])
          that returns a tf.keras.callbacks.History object.
        :param log_dir: Top level log directory where tensorboard run info is stored
        :param args: Args to keras_tuner.Tuner subclass
        :param kwargs: Kwargs to keras_tuner.Tuner subclass
        :param objective_metric: The metric in history object to use as objective metric

        As this subclass overrides run_trial, the hypermodel argument is invalid.
        """
        self._logger = get_logger(self.__class__.__name__, 'info')
        self._logger.info(f'Log dir {log_dir}')
        directory = kwargs.pop('directory', log_dir)
        kwargs.update({'directory': directory})
        project_name = kwargs.pop('project_name', self.__class__.__name__)
        kwargs.update({'project_name': project_name})
        super().__init__(*args, **kwargs)
        self._build_fn = build_fn
        self._fit_fn = fit_fn
        self._log_dir = log_dir
        self._objective_metric = objective_metric

    def get_trial_log_dir(self, trial: keras_tuner.engine.trial.Trial) -> str:
        # Helper function to generate a log dir path that matches with keras_tuner.oracle logging
        return self._log_dir + f'/{self.__class__.__name__}/trial_{trial.trial_id}'

    def run_trial(self, trial: keras_tuner.engine.trial.Trial, *args, **kwargs) -> float:
        # Unused args, kwargs
        del args
        del kwargs

        model = self._build_fn(hparams=trial.hyperparameters)

        if len(trial.hyperparameters.values.keys()) == 0:
            raise ValueError('Expected that some hyperparameters were set, but got none.')

        # Setup tensorboard and hyperparameter logging from keras_tuner
        # Run the config tensorboard after the model is built, so that hparams scope is defined (by build_model)
        # and then pass on the callbacks to the model fit call.
        if keras_tuner_backend.config.backend() != "tensorflow":
            # Below code requires that Tensorflow is the backend (excerpt from super class)
            raise ValueError('Required tensorflow backend is not used.')
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=self.get_trial_log_dir(trial))]  # modified by _configure_tensorboard_dir
        hparams = tuner_utils.convert_hyperparams_to_hparams(
            trial.hyperparameters,
            hparams_api,
        )
        callbacks.append(
            hparams_api.KerasCallback(
                writer=self.get_trial_log_dir(trial),
                hparams=hparams,
                trial_id=trial.trial_id
            )
        )
        history: tf.keras.callbacks.History = self._fit_fn(model, tuning_callbacks=callbacks)
        metric: list = history.history[self._objective_metric]
        return metric[-1]

    def search(self, *args, **kwargs):
        """
        Providing args and kwargs to model.fit() via search() is not supported as
        run_trial() overrides self.hypermodel behavior.
        """
        del args
        del kwargs
        super().search()


class GridSearchTuner(CustomTuner, keras_tuner.GridSearch):
    pass


class BayesianTuner(CustomTuner, keras_tuner.BayesianOptimization):
    pass


class RandomSearchTuner(CustomTuner, keras_tuner.RandomSearch):
    pass

# FIXME: Does not run as expected
#class HyperbandTuner(CustomTuner, keras_tuner.Hyperband):
#    pass
