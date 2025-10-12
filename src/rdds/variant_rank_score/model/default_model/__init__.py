import os
from .model_file_spec import ModelFileSpec

DEFAULT_MODEL_VERSION = "v1.12.0-rc1-10-g0a7de5f"
DEFAULT_MODEL_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_VERSION))
DEFAULT_KERAS_MODEL_FILE = os.path.join(DEFAULT_MODEL_DIRECTORY, 'saved-models', '34-0.0064.keras')
DEFAULT_VOCABULARY_FILE = os.path.join(DEFAULT_MODEL_DIRECTORY, 'vocabulary.txt')
DEFAULT_NUMERICAL_NORMALISATION_WEIGHTS = os.path.join(DEFAULT_MODEL_DIRECTORY, 'normalisation.tar')
DEFAULT_MODEL_EXPLAINER = os.path.join(DEFAULT_MODEL_DIRECTORY, 'model-explainer.bin')
DEFAULT_MODEL_SPEC = ModelFileSpec(
    model_version=DEFAULT_MODEL_VERSION,
    keras_model=DEFAULT_KERAS_MODEL_FILE,
    vocabulary_file=DEFAULT_VOCABULARY_FILE,
    numerical_normalisation_weights=DEFAULT_NUMERICAL_NORMALISATION_WEIGHTS,
    explainer_model=DEFAULT_MODEL_EXPLAINER,
)
