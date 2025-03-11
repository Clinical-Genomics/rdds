import tensorflow as tf
from typing import Tuple, Union

# Enable to debug tf.function
#tf.config.run_functions_eagerly(True)
#tf.autograph.set_verbosity(1, alsologtostdout=True)


class AugmentDropoutDataset:

    def __init__(self,
                 target_data_tensor_idx: int,
                 dropout_on_categorical_label_value: float,
                 seed: int,
                 dropout_ratio: float,
                 dropout_value: Union[bytes, float]):
        """
        :param target_data_tensor_idx: Index of Tensor in data tuple (tensor0, ...)
        :param dropout_on_categorical_label_value: A categorical class value (precondition) for dropout to occur
        :param seed: Seed for RNG. For uncorrelated augmentation across tensors, make sure to set
            separate seeds for each tensor.
        :param dropout_value: The replacement value (when augmented)
        :param dropout_ratio: The probability of replacement (augmentation), [0, 1]
        """
        self._target_data_tensor_idx = target_data_tensor_idx
        self._dropout_on_categorical_label_value = tf.constant(dropout_on_categorical_label_value, dtype=tf.float32)
        self._seed = seed
        with tf.device('/CPU:0'):
            # RNG must run on CPU since tf.op not implemented for GPU
            self._rng: tf.random.Generator = tf.random.Generator.from_seed(seed=seed)
        self._dropout_ratio: tf.Tensor = tf.constant([dropout_ratio], dtype=tf.float32)
        if isinstance(dropout_value, bytes):
            dtype = tf.string
        elif isinstance(dropout_value, float):
            dtype = tf.float32
        else:
            raise ValueError(f'Unsupported dtype {type(dropout_value)}')
        self._dropout_value: tf.Tensor = tf.constant(dropout_value, dtype=dtype)

    @tf.function
    def _apply_dropout(self,
                       input: tf.Tensor,
                       pathogenic_label: tf.Tensor) -> tf.Tensor:
        """
        Apply dropout to tensor replacing input with self._dropout_value
        :param input: Input tensor to dropout
        :param pathogenic_label: Tensor containing categorical label, [0.0, 1.0]
            A value of 1.0 will apply dropout with probability self._dropout_ratio.
        :return: Dropped out tensor with shape of input
        """
        do_dropout = self._rng.binomial(shape=tf.constant([1]),
                                        counts=tf.constant([1.], dtype=tf.float32),
                                        probs=self._dropout_ratio  # prob: of success (i.e. do dropout)
                                        )
        do_dropout = do_dropout[0]  # flatten
        do_dropout = tf.cast(do_dropout, tf.float32)
        dropped_out_input: tf.Tensor = tf.fill(dims=tf.shape(input), value=self._dropout_value)  # Replacement tensor
        pred_dropout: tf.Tensor = tf.greater(do_dropout,
                                             tf.constant(0, dtype=tf.float32))  # if boolean == true
        pred_is_pathogenic: tf.Tensor = tf.equal(pathogenic_label,
                                                 self._dropout_on_categorical_label_value)
        pred = tf.logical_and(pred_dropout, pred_is_pathogenic)
        output = tf.cond(pred=pred,
                         true_fn=lambda: dropped_out_input,
                         false_fn=lambda: input)
        return output

    def _process(self, data_tensors, labels) -> Tuple[Tuple[tf.Tensor, ...], ...]:
        """
        :param data_tensors: A non-nested tuple of tensors, (tensor0, ...)
        :param labels: A tuple (label_tensor, ) containing categorical labels of shape [2]
        :return: (tuple of data tensors, tuple of label)
        """
        label_tensor, = labels  # unpack tuple (tensor, )
        label_pathogenic = label_tensor[1]  # select pathogenic label [benign, pathogenic] class
        # Assume tensor tuples are ordered in same order as element_spec
        output_tensors: Tuple[tf.Tensor] = tuple()
        if self._target_data_tensor_idx > len(data_tensors):
            raise ValueError(f'Attempted to process index out of bounds {self._target_data_tensor_idx}:{data_tensors}')
        for idx, tensor in enumerate(data_tensors):
            if idx == self._target_data_tensor_idx:
                dropped_out_tensor = self._apply_dropout(input=tensor,
                                                         pathogenic_label=label_pathogenic)
                output_tensors += (dropped_out_tensor,)
            else:
                output_tensors += (tensor,)
        output = (output_tensors, labels)
        return output

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Only accepts rows of input (batches invalid)
        :param dataset:
        :return:
        """
        dataset = dataset.map(self._process,
                              num_parallel_calls=tf.data.AUTOTUNE,
                              name=f'map_{self._target_data_tensor_idx}')
        return dataset


class TextAugmentDropoutDataset(AugmentDropoutDataset):

    def __init__(self, dropout_value=b'', *args, **kwargs):
        super().__init__(*args, dropout_value=dropout_value, **kwargs)


class NumericalAugmentDropoutDataset(AugmentDropoutDataset):

    def __init__(self, dropout_value=0.0, *args, **kwargs):
        super().__init__(*args, dropout_value=dropout_value, **kwargs)