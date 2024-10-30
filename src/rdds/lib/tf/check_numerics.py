import tensorflow as tf

from rdds.lib.logging import get_logger


def enable_check_numerics():
    """
    Enable numnerical NaN and +-Inf checks in model during training and inference.

    Due to missing support in XLA library for check_numerics API, running
    NaN checks at GPU is unsupported.

    See ticket: https://github.com/tensorflow/tensorflow/issues/59215
    """
    logger = get_logger('check_numerics', 'info')
    gpu_devices = tf.config.list_physical_devices('GPU')
    if len(gpu_devices) > 0:
        logger.warning(f'check_numerics disabled due to GPU available: {gpu_devices}')
    else:
        tf.debugging.enable_check_numerics()
