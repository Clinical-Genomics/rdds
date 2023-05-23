from rdds.exploration_rankscore.confusion_matrix import ConfusionMatrix
from rdds.exploration_rankscore.roc import compute_roc


def test_roc():
    """
    Test Receiver Operating Characteristic computation
    :return:
    """
    # GIVEN some confusion matrix
    # WHEN computing the ROC metric
    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,  # TP test
                                              true_negatives=0,
                                              false_positives=0,
                                              false_negatives=0)
    roc_metric_coordinate = compute_roc(matrix)
    # THEN expect proper values
    assert roc_metric_coordinate.false_positive_rate == 0.0, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 1.0, roc_metric_coordinate

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=1,  # TN test
                                              false_positives=0,
                                              false_negatives=0)
    roc_metric_coordinate = compute_roc(matrix)
    assert roc_metric_coordinate.false_positive_rate == 0.0, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 0.0, roc_metric_coordinate

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_positives=1,  # FP test
                                              false_negatives=0)
    roc_metric_coordinate = compute_roc(matrix)
    assert roc_metric_coordinate.false_positive_rate == 1.0, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 0.0, roc_metric_coordinate

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=0,
                                              true_negatives=0,
                                              false_positives=0,
                                              false_negatives=1)  # FN test
    roc_metric_coordinate = compute_roc(matrix)
    assert roc_metric_coordinate.false_positive_rate == 0.0, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 0.0, roc_metric_coordinate

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,  # Mixed case
                                              true_negatives=1,
                                              false_positives=0,
                                              false_negatives=0)
    roc_metric_coordinate = compute_roc(matrix)
    assert roc_metric_coordinate.false_positive_rate == 0.0, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 1.0, roc_metric_coordinate

    matrix: ConfusionMatrix = ConfusionMatrix(true_positives=1,
                                              true_negatives=1,
                                              false_positives=1,
                                              false_negatives=1)  # Mixed case
    roc_metric_coordinate = compute_roc(matrix)
    assert roc_metric_coordinate.false_positive_rate == 0.5, roc_metric_coordinate
    assert roc_metric_coordinate.true_positive_rate == 0.5, roc_metric_coordinate
