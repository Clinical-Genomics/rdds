from dataclasses import dataclass
import os


@dataclass
class ModelFileSpec:
    model_version: str
    keras_model: str

DEFAULT_MODEL_VERSION = "20251021-203805-gicam-improvement-rc4"
DEFAULT_MODEL_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_VERSION))
DEFAULT_KERAS_MODEL_FILE = os.path.join(DEFAULT_MODEL_DIRECTORY, 'saved-models', '50-0.0005.keras')
DEFAULT_MODEL_SPEC = ModelFileSpec(
    model_version=DEFAULT_MODEL_VERSION,
    keras_model=DEFAULT_KERAS_MODEL_FILE
)