from dataclasses import dataclass


@dataclass
class ModelFileSpec:
    model_version: str
    keras_model: str
    explainer_model: str
    vocabulary_file: str
    numerical_normalisation_weights: str
