import pandas as pd
import numpy as np
from typing import Tuple

from rdds.lib.logging import get_logger
from ..dataset import SET_TEST, SET_TRAIN

LOGGER = get_logger("DatasetLoader", "info")


class DatasetLoader:
    def __init__(self, path_to_dataset: str):
        self._df = pd.read_csv(path_to_dataset, low_memory=False)
        LOGGER.info(f"Loaded {len(self._df)} samples from {path_to_dataset}")
        print(self._df)

    def _get_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]], set_type: int
    ):
        data_spec, label_spec = input_spec
        df = self._df[self._df.set == set_type]
        x = df[list(data_spec)].values
        y = df[list(label_spec)].values
        return x, y

    def get_train_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]]
    ) -> Tuple[np.ndarray]:
        return self._get_data(input_spec=input_spec, set_type=SET_TRAIN)

    def get_test_data(
        self, input_spec: Tuple[Tuple[str, ...], Tuple[str, ...]]
    ) -> Tuple[np.ndarray]:
        return self._get_data(input_spec=input_spec, set_type=SET_TEST)
