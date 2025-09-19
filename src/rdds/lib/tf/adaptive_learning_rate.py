def adaptive_learning_rate(network_param: int,
                           epoch_number: int,
                           warmup_epochs: int):
    """
    Adaptive learning rate.

    Linearly increasing until warmup_epochs and then
    decreasing proportionally to the inverse of the square root
    of the step number.
    :param network_param: A scaling value (often depends on network param)
    :param epoch_number: Epoch step number
    :param warmup_epochs: Amount of warmup epochs
    """

    if epoch_number == 0:
        epoch_number = 1

    lr = (network_param ** -0.5) * \
         min(epoch_number ** -0.5, epoch_number * warmup_epochs ** -1.5)
    return lr