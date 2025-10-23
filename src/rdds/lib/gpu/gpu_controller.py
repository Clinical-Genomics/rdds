from os import environ
from rdds.lib.logging import get_logger
_LOGGER = get_logger('gpu_controller', 'info')

ENV_NAME = 'CUDA_VISIBLE_DEVICES'
_NOT_SET = 'NO_CONFIG_SET_IN_ENVIRONMENT'

class GpuController:

    """
    Helper class to control NVIDIA CUDA enabled GPUs.
    """

    def __init__(self):
        try:
            self._default_state = environ[ENV_NAME]
        except KeyError:
            self._default_state = _NOT_SET

    @property
    def default_state(self) -> str:
        return self._default_state

    @property
    def disabled_state(self) -> str:
        return ''

    def disable_gpus(self):
        _LOGGER.info('Disabling GPUs')
        environ[ENV_NAME] = self.disabled_state

    def restore_defaults(self):
        if self._default_state == _NOT_SET:
            environ.unsetenv(ENV_NAME)
        else:
            environ[ENV_NAME] = self.default_state
        _LOGGER.info(f'Restored default state: {self._default_state}')