from rdds.lib.hpt import HyperParameters, CustomTuner, GridSearchTuner, RandomSearchTuner, BayesianTuner
from ..model import Gicam


class GicamTuner(CustomTuner):

    def __init__(self,
                 hd5_file_path: str,
                 max_epochs: int,
                 *args,
                 amount_data=0.025,
                 **kwargs):
        self._hd5_file_path = hd5_file_path
        self._max_epochs = max_epochs
        self._amount_data = amount_data
        # Provide class methods as the build and fit methods.
        kwargs.update({'build_fn': self._build_model_fn})
        kwargs.update({'fit_fn': self._fit_fn})
        # Target metric
        kwargs.update({'objective_metric_fn': self._maximize_mcc_metric})
        kwargs.update({'objective_metric': None})  # Handled by objective_metric_fn
        super().__init__(*args, **kwargs)

    @staticmethod
    def _maximize_mcc_metric(history: dict) -> float:
        # Invert MCC to allow MAX(MCC)
        mccs = history['val_MCC']
        return 1.0 - mccs[-1]

    def _build_model_fn(self,
                        hparams: HyperParameters,
                        trial_work_dir: str) -> Gicam:
        model = Gicam(work_dir=trial_work_dir,
                      train_max_epochs=self._max_epochs)
        model.build(path_to_hd5_dataset=self._hd5_file_path,
                    hparams=hparams,
                    amount_data=self._amount_data)
        return model

    @staticmethod
    def _fit_fn(model: Gicam,
                tuning_callbacks):
        return model.train(hparam_tuning_callbacks=tuning_callbacks, validation_only_beginning_end=True)


class GicamGridSearchTuner(GicamTuner, GridSearchTuner): pass


class GicamRandomSearchTuner(GicamTuner, RandomSearchTuner): pass


class GicamBayesianTuner(GicamTuner, BayesianTuner): pass