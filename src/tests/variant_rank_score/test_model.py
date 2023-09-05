import tensorflow as tf
import pytest as pt
import h5py
import os
import numpy as np

from rdds.variant_rank_score.model import VariantRankScoreModel


@pt.fixture
def hd5_file_path() -> str:
    hd5_file_path = '/tmp/vrsmodel-test.hd5'
    hd5_file = h5py.File(hd5_file_path, 'w')
    group = hd5_file.create_group(name='train')
    dataset = group.create_dataset(name='feature_numerical_a', dtype=float, shape=(2, ))
    dataset[:] = np.arange(0, 2)
    dataset = group.create_dataset(name='feature_textual_a', dtype=h5py.string_dtype(), shape=(2, ))
    dataset[:] = [b'hello', b'world']
    dataset = group.create_dataset(name='feature_textual_b', dtype=h5py.string_dtype(), shape=(2, ))
    dataset[:] = [b'foo', b'bar']
    dataset = group.create_dataset(name='label', dtype=float, shape=(2, ))
    dataset[:] = [0.0, 1.0]
    hd5_file.copy('train', 'test')
    hd5_file.flush()
    hd5_file.close()
    yield hd5_file_path
    os.remove(hd5_file_path)


def test_model_train(hd5_file_path):
    """
    Test for model train and predict.
    """
    model: VariantRankScoreModel = VariantRankScoreModel(features_text=['feature_textual_a', 'feature_textual_b'],
                                                         features_numerical=['feature_numerical_a'])
    model.train(hd5_file_path=hd5_file_path)
    input_data = {'input_text': np.array([['hello foo', 'bar']]),
                  'input_numerical': np.array([[0.0]])}
    y = model.predict(input_data=input_data)
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 2)
    assert np.max(y) < 1
    assert np.min(y) > 0
