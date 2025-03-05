from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.eager.profiler import maybe_create_event_file as legacy_tensorboard_event_file_creator
from tensorflow.python.profiler.trace import Trace

from rdds.lib.logging import get_logger
_LOGGER = get_logger(name='tf-profiler', log_level='info')

DEFAULT_PROFILER_OPTIONS = profiler.ProfilerOptions(host_tracer_level=3,
                                                    python_tracer_level=1,
                                                    device_tracer_level=1)

class TfProfiler:

    """
    Helper class to run profiling of TF graps.

    Use this class in conjunction with Trace().

    NOTE: It's undefined to run both keras.tensorboard callback profiling and TfProfiler simultaneously.

    NOTE: The Keras TensorBoard callback will automatically perform sampled
    profiling. Before enabling customized profiling, set the callback flag
    "profile_batches=[]" to disable automatic sampled profiling.

    NOTE: on GPU prerequisites: https://www.tensorflow.org/guide/profiler#install_the_profiler_and_gpu_prerequisites

    https://www.tensorflow.org/guide/profiler#profiling_custom_training_loops
    """

    def __init__(self, logdir: str):
        self._logdir = logdir
        self._started = False

    def start_if_not_running(self, *args, **kwargs):
        if self._started:
            return
        self.start(*args, **kwargs)

    def start(self, default_profiler_options: profiler.ProfilerOptions = DEFAULT_PROFILER_OPTIONS):
        """
        Immediately start to profile once this is called, regardless whether a Trace is created or not.
        Make sure to call this at the appropriate time to get the intended profiling results
        (profiler can only support a limited amount of traces (further limited by tracing depth).
        :param default_profiler_options:
        :return:
        """
        _LOGGER.info(f'Profiling logs saved in {self._logdir}')
        # If event file is not created, Tensorboard won't show profiling results
        legacy_tensorboard_event_file_creator(self._logdir)
        profiler.start(self._logdir, options=default_profiler_options)
        self._started = True

    def stop(self):
        profiler.stop()
        msg = (f'\
Profiling complete. View results by: \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -m tensorboard.main  --port 4000 --logdir {self._logdir}\n \
firefox http://localhost:4000#profile')
        _LOGGER.info(msg)
