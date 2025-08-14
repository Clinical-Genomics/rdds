import pytest as pt
import numpy as np
import os
from rdds.lib.list_dir import list_dir
import tensorflow as tf

from rdds.gicam import Gicam


@pt.fixture
def new_keras_session():
    tf.keras.backend.clear_session()


def test_model_init(new_keras_session):
    """
    Test for initializing a model instance
    """
    model = Gicam()


def test_model_train(new_keras_session):
    """
    Test for training mechanics
    """
    model = Gicam()
    model.train(path_to_dataset="/rdds/src/tests/gicam/test_data.csv")


def test_load_model(new_keras_session):
    """
    Test for training and restoring model from a saved model file
    """
    model = Gicam()
    model.train(path_to_dataset="/rdds/src/tests/gicam/test_data.csv")
    files = list_dir(model._train_log_dir)
    saved_model_file = [file for file in files if ".keras" in file][0]
    del model
    tf.keras.backend.clear_session()
    Gicam.from_saved_model(saved_model_file)
