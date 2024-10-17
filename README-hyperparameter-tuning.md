# Hyperparameter Tuning

Questions:
1. Can we make use of tensorboard and keras tuner at the same time?
	- use the hparams_config and summary_writer in tensorboard
	See: https://keras.io/guides/keras_tuner/visualize_tuning/

## Keras/kerastuner
Advantage here is that you can use different algos to
traverse the hpt space in a more efficient manner
than using a grid search.

For reproducibility, use the GridSearch tuner.

https://keras.io/guides/keras_tuner/getting_started/

Source is here: https://github.com/keras-team/keras-tuner

### Separating model and kerastuner
This is good for readability and reproducibility.
https://keras.io/guides/keras_tuner/getting_started/#keep-keras-code-separate

## Tensorflow/hparams
Grid search style only which makes this technique reproducible.

Part of tensorflow bundle and built to work with tensorboard.

API in tensorflow and tensorboard is still in beta stage, so
expect breaking changes.

https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

## Tensorflow Decision Tree Tuner (TF-DFs tuner) and kerastuner
Seems like this is targetin the TF/DecisionTree package and not
Tensorflow/tensorflow in general.

https://www.tensorflow.org/decision_forests/tutorials/automatic_tuning_colab
