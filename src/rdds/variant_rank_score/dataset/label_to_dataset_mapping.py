from typing import Dict
from enum import IntEnum


class DATASET_TYPE(IntEnum):
    TRAIN = 0
    TEST = 1
    BOTH = 2


LABEL_TO_DATASET_MAPPING: Dict[str, DATASET_TYPE] = {
    'CLINVAR': DATASET_TYPE.TRAIN,
    'GIAB': DATASET_TYPE.BOTH,
    'MUTACC': DATASET_TYPE.TEST
}