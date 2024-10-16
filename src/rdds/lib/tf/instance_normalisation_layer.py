import tensorflow as tf
from typing import List
import numpy as np
import tempfile
import os
import tarfile
from datetime import datetime
import logging

from rdds.lib.list_dir import list_dir
from rdds.lib.logging import get_logger

_LOGGER = get_logger('InstanceNormalisation')
_LOGGER.setLevel(logging.INFO)


@tf.keras.saving.register_keras_serializable()  # Make sure layer is available in keras save/ load operations.
class InstanceNormalisationLayer(tf.keras.layers.Normalization):

    """
    Layer that performs normalisation on per feature column.
    """

    def __init__(self,
                 *args,
                 axis=-1,
                 **kwargs):
        super().__init__(*args, axis=axis, **kwargs)

    def adapt_from_dataset(self,
                           data: tf.data.Dataset):
        """
        Adapt layer weights, mean, variance to dataset.

        NOTE: Parent method does not accept tf.data.Datasets with nested signatures;
        (TensorSpec(...), ). Make sure to provide a tf.data.Dataset with
        a flat TensorSpec, TensorSpec(...).
        """
        if not isinstance(data.element_spec, tf.TensorSpec):
            raise ValueError(f'Provide a single TensorSpec, got {data.element_spec}')
        time_start: datetime = datetime.now()
        super().adapt(data=data)
        _LOGGER.info(f'Normalisation parameter adapt took {datetime.now() - time_start}')

    def save_weights_to_file(self, file_path: str):
        """
        Save layer weights to tar archive.
        This tar archive can be used to re-init layer to avoid calling adapt_from_dataset().
        :param file_path: Path to file
        """
        tmp_dir = tempfile.TemporaryDirectory()
        weights: List[np.ndarray] = self.get_weights()
        for weight_idx, weight in enumerate(weights):
            weight_file_path: str = os.path.join(tmp_dir.name, f'{weight_idx}.npy')
            np.save(file=weight_file_path, arr=weight)
        tar_archive = tarfile.open(name=file_path, mode='w')
        tar_archive.add(name=tmp_dir.name)
        tar_archive.close()
        _LOGGER.info(f'Stored normalisation weights to tar archive: {file_path}')

    def load_saved_weights_file(self, file_path: str):
        """
        Load weights from stored weights file.
        Make sure layer is called (example; y = layer(input)) prior to calling this method,
        with same shape on input.
        :param file_path: The tar archive containing weights, stored with save_weights_to_file()
        :raises ValueError: In case failing to extract weights from archive
        """
        tmp_dir = tempfile.TemporaryDirectory()
        tar_archive = tarfile.open(file_path, 'r')
        tar_archive.extractall(path=tmp_dir.name)
        tar_archive.close()
        weight_files: List[str] = list(list_dir(tmp_dir.name))
        if len(weight_files) == 0:
            raise ValueError('Expected some weight files but got none')
        # Sort files in increasing weight name order
        weight_files.sort(key=lambda file_path: file_path.split('/')[-1], reverse=False)
        loaded_weights: List[np.ndarray] = []
        for weight_file in weight_files:
            loaded_weights.append(np.load(weight_file))
        self.set_weights(loaded_weights)
        _LOGGER.info(f'Loaded weights from {file_path}')
        _LOGGER.info(f'Normalisation weights:\n{self.weights}')
