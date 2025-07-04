import tensorflow as tf
from typing import Tuple

from ..dataset import DatasetLoader

from rdds.variant_rank_score.model.custom_metrics import MccScore, F1Score

"""
Static metrics that will once computed once, will persist as as static value across all epochs during trainig.

Used for computing a baseline to compare model improvement against.
"""

class StaticF1Metric(F1Score):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass


class StaticMccMetric(MccScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticTruePositives(tf.keras.metrics.TruePositives):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticTrueNegatives(tf.keras.metrics.TrueNegatives):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticFalseNegatives(tf.keras.metrics.FalseNegatives):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticPrecision(tf.keras.metrics.Precision):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticRecall(tf.keras.metrics.Recall):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticFalsePositives(tf.keras.metrics.FalsePositives):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

class StaticAUC(tf.keras.metrics.AUC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._computed = False

    def update_state(self, *args, **kwargs):
        if not self._computed:
            super().update_state(*args, **kwargs)
            self._computed = True

    def reset_state(self):
        pass

def compute_performance_baseline(dataset_loader: DatasetLoader) -> Tuple[tf.keras.metrics.Metric, ...]:

    """
    Compute baseline metrics for model evaluation purposes.
    Use MIVMIR as reference.

    TODO: Add baseline metric as to F1, MCC main metrics (as _train, _validation subset) instead of
    as now (separate metrics).
    """

    input_spec = ("score_mivmir", ), ("pathogenic",)
    x_train, y_train = dataset_loader.get_train_data(input_spec=input_spec)
    x_test, y_test = dataset_loader.get_test_data(input_spec=input_spec)

    def setup_metrics(y, x, set_name) -> tuple:
        recall = StaticRecall(name=f'MIVMIR_recall_{set_name}_baseline')
        recall.update_state(y[:, 0],
                            x[:, 0])
        precision = StaticPrecision(name=f'MIVMIR_precision_{set_name}_baseline')
        precision.update_state(y[:, 0],
                               x[:, 0])
        f1 = StaticF1Metric(name=f'MIVMIR_F1_{set_name}_baseline')
        f1.update_state(y=y[:, 0],
                        y_pred=x[:, 0])
        mcc = StaticMccMetric(name=f'MIVMIR_MCC_{set_name}_baseline')
        mcc.update_state(y=y[:, 0],
                         y_pred=x[:, 0])
        tp = StaticTruePositives(name=f'MIVMIR_true_positives_{set_name}_baseline')
        tp.update_state(y[:, 0],
                        x[:, 0])
        tn = StaticTrueNegatives(name=f'MIVMIR_true_negatives_{set_name}_baseline')
        tn.update_state(y[:, 0],
                        x[:, 0])
        fp = StaticFalsePositives(name=f'MIVMIR_false_positives_{set_name}_baseline')
        fp.update_state(y[:, 0],
                        x[:, 0])
        fn = StaticFalseNegatives(name=f'MIVMIR_false_negatives_{set_name}_baseline')
        fn.update_state(y[:, 0],
                        x[:, 0])
        auc = StaticAUC(name=f'MIVMIR_auc_{set_name}_baseline')
        auc.update_state(y[:, 0],
                         x[:, 0])
        return precision, recall, f1, mcc, tp, tn, fp, fn, auc

    all_metrics = tuple()
    all_metrics += setup_metrics(y=y_train, x=x_train, set_name='train')
    all_metrics += setup_metrics(y=y_test, x=x_test, set_name='test')
    return all_metrics