import pytest as pt
import numpy as np
import os
from rdds.lib.list_dir import list_dir
import tensorflow as tf
from h5py import File as Hd5File, string_dtype, Dataset as Hd5DataSet, Group as Hd5Group
import pandas as pd

from rdds.lib.hpt import HyperParameters
from rdds.gicam.model import Gicam

@pt.fixture
def path_to_hd5_dataset(work_dir):
    df = pd.read_csv("/rdds/src/tests/gicam/test_data.csv")
    file_path = os.path.join(work_dir, 'dataset.hd5')
    hd5_file: Hd5File = Hd5File(file_path, 'w')
    hd5_file.create_group('gicamdata')
    hd5_file.create_dataset('gicamdata/mivmir',
                            dtype=np.float32,
                            data=df.score_mivmir)
    hd5_file.create_dataset('gicamdata/genmod',
                            dtype=np.float32,
                            data=df.score_genmod)
    hd5_file.create_dataset('gicamdata/causative',
                            dtype=np.float32,
                            data=df.causative)
    hd5_file.flush()
    hd5_file.close()
    yield file_path

@pt.fixture
def new_keras_session():
    tf.keras.backend.clear_session()


def test_model_init(new_keras_session):
    """
    Test for initializing a model instance
    """
    model = Gicam()


def test_model_train(new_keras_session, path_to_hd5_dataset):
    """
    Test for training mechanics
    """
    model = Gicam(train_max_epochs=5)
    model.build(path_to_hd5_dataset=path_to_hd5_dataset,
                hparams=HyperParameters())
    model.train()


def test_load_model(new_keras_session, path_to_hd5_dataset):
    """
    Test for training and restoring model from a saved model file
    """
    model = Gicam(train_max_epochs=1)
    model.build(path_to_hd5_dataset=path_to_hd5_dataset,
                hparams=HyperParameters())
    model.train()
    files = list_dir(model._train_log_dir)
    saved_model_file = [file for file in files if ".keras" in file][0]
    del model
    tf.keras.backend.clear_session()
    Gicam.from_saved_model(saved_model_file)

def test_load_default_model(new_keras_session, work_dir):
    """
    Test for training and restoring model from a saved model file.
    Run some inference (and generate a plot)
    """
    gicam = Gicam.from_saved_model()
    gicam.visualize_decision_boundary(storage_path=work_dir)
