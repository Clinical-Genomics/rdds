import tensorflow as tf


class EpochLogger(tf.keras.callbacks.ProgbarLogger):
    """
    Logger of various steps in model training.

    Append to callbacks in fit()
    """
    def __init__(self, logger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = logger

    def on_train_begin(self, *args, **kwargs):
        self._logger.info('training')
        super().on_train_begin(*args, **kwargs)

    def on_test_begin(self, *args, **kwargs):
        self._logger.info('testing')
        super().on_test_begin(*args, **kwargs)

    def on_train_batch_begin(self, *args, **kwargs):
        self._logger.info(f"train batch {args}, {kwargs}")
        super().on_train_batch_begin(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        self._logger.info(f"train batch complete {args}, {kwargs}")
        super().on_train_batch_end(*args, **kwargs)

    def on_test_batch_begin(self, *args, **kwargs):
        self._logger.info(f"test batch {args}, {kwargs}")
        super().on_test_batch_begin(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self._logger.info(f"test batch complete {args}, {kwargs}")
        super().on_test_batch_end(*args, **kwargs)
