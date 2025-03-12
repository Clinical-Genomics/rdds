from rdds.lib.hpt import HyperParameters, CustomTuner, GridSearchTuner, RandomSearchTuner, BayesianTuner
from rdds.variant_rank_score.model import VariantRankScoreModel


class VRSModelTuner(CustomTuner):

    def __init__(self, hd5_file_path: str, *args, **kwargs):
        self._hd5_file_path = hd5_file_path
        # Provide class methods as the build and fit methods.
        kwargs.update({'build_fn': self._build_model_fn})
        kwargs.update({'fit_fn': self._fit_fn})
        super().__init__(*args, **kwargs)

    def _build_model_fn(self,
                        hparams: HyperParameters,
                        trial_work_dir: str) -> VariantRankScoreModel:
        model = VariantRankScoreModel(workdir=trial_work_dir,
                                      workdir_suffix='')
        model.build(hd5_file_path=self._hd5_file_path,
                    hparams=hparams,
                    compile_vocabulary_normalisation_factors=False)
        return model

    @staticmethod
    def _fit_fn(model: VariantRankScoreModel,
                tuning_callbacks):
        return model.train(hparam_tuning_callbacks=tuning_callbacks)


class VRSGridSearchTuner(VRSModelTuner, GridSearchTuner): pass


class VRSRandomSearchTuner(VRSModelTuner, RandomSearchTuner): pass


class VRSBayesianTuner(VRSModelTuner, BayesianTuner): pass