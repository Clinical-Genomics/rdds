# Tensorflow Profiler

This tool is used to view performance of TF graphs and pipelines.

## Tensorboard Dependencies
Pyenv must have the `tensorboard-plugin-profile==2.13.1` installed
in order to view profiling results.

## GPU Limitations
Profiling on GPUs requires the Nvidia CUPTI library, if this is not
present, profiling will fail.

If CUPTI lib is not available, one can disable GPUs temporarily
by setting environment variable `CUDA_VISIBLE_DEVICES=""`.

## Tracing

One must use the `Trace` context manager to record profiling
traces for actions.

```python
dataset: tf.data.Dataset
profiler = TfProfiler(logdir=logdir)
profiler.start()
dataset = dataset.__iter__()
for step in range(10):
    with Trace('batch', step_num=step, _r=1):
       _ = next(dataset)
profiler.stop()
```

> `_r` keyword argument makes this trace event get processed as a step event by the Profiler.
