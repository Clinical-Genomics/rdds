# Hyperparameter Tuning Module

This module helps in hyperparameter optimisation.

For current up-to date usage of this module, refer
to the relevant tests.

## Hyperparameter API
See https://keras.io/api/keras_tuner/hyperparameters

## Limiting search space
```python
# Define search space
hparams = Hyperparameters()
hparams.Int(...)

# tune_new_entries forces defaults for later created hparams
tuner = Tuner(..., hyperparameters=hparams, tune_new_entries=False)
tuner.search()
``
