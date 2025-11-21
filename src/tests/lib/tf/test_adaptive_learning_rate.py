from rdds.lib.tf.adaptive_learning_rate import _adaptive_learning_rate


def test_learning_rate():
    ys = []
    xs = list(range(0, 100))
    for x in xs:
        y = _adaptive_learning_rate(network_param=512,
                                   epoch_number=x,
                                   warmup_epochs=10)
        ys.append(y)
