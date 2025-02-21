# Debug Tensorflow Processing

https://www.tensorflow.org/guide/profiler#trace_viewer

## Dependencies
The profiling tool is somewhat unstable and does not work out of the box.
Only certain versions seems to be stable, and one has to disable protobuf parsing in
tensorboard to parse the profiling results.

Install `tensorflow-plugin-profile` as pip requirement (not installed by default).


## Enable profiling logging
Note that the batch limits '(10,15)' are the total batch index across all datasets (train, test)
incrementing during training and testing step execution.
The tuple must be specified exactly as above, no space after the comma.

```
callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self._train_log_dir,
                                                histogram_freq=1,
                                                profile_batch='(10,15)', <<<< debug batches 10 -> 15
                                                embeddings_freq=1)) 
```

Note that during the first epoch, dataset is most likely caching data so performance is poor in data io step.

## View profiling results in Tensorboard
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 -m tensorboard.main  --port 4000 --logdir [LOGDIR]`
