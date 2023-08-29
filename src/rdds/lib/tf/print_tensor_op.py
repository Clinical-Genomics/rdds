from typing import Union
import keras.engine.keras_tensor
import tensorflow as tf

TensorTypes = Union[tf.Tensor, tf.RaggedTensor, keras.engine.keras_tensor.KerasTensor]


def print_tensor_op(tensor: TensorTypes,
                    message: str = '',
                    summarize: int = -1) -> TensorTypes:
    """
    Prints a Tensor or KerasTensor when executed inside the computational graph.
    :param tensor: The tensor to be printed
    :param message: A text message
    :param summarize: -1 prints content of all tensor, otherwise n samples
    :return: The printed tensor, make sure to use it in the graph to have this method called.
    """
    tensor = tf.keras.backend.print_tensor(tensor, message=message, summarize=summarize)
    return tensor
