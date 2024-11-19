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

## Known Issues
### hparams.Conditional_scope and Scoped Hyperparameters
Using sub-hparams in a conditional_scope is prone to errors like
`ValueError: multiple values specified for hparam '[HPT_NAME]'` if using the hyperparameter scope limitation:
```python
Tuner(hparams=limited_hparams, tune_new_entries=False)
```
if hparams is created like this in a model:
```python
feature_flag = hparams.Boolean('feature_a', default=True)
with hparams.conditional_scope('feature_a', [True]):
    hparams.Float(....)
```
