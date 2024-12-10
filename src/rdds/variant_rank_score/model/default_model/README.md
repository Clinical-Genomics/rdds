# Default Model
This directory contains the default model, in case not overridden
by user.

The model directories are versioned by sematic versioning,
`v[VERSION]`.

## Updating the directory - Adding a new model

> Beware of the Github maximum file size of 25MB.

Example layout:
```raw
.
├── build-config.txt
├── compile-config.txt
├── dataset-config.txt
├── hyperparams.txt
├── network-config.txt
├── normalisation.tar
├── projector_config.pbtxt
├── saved-models
│   └── 15-0.0005.keras
├── train
│   ├── checkpoint
│   ├── events.out.tfevents.1730795975.gpu-compute-0-0.local.17.0.v2
│   ├── keras_embedding.ckpt-15.data-00000-of-00001
│   └── keras_embedding.ckpt-15.index
├── validation
│   └── events.out.tfevents.1730797570.gpu-compute-0-0.local.17.1.v2
└── vocabulary.txt
```

### Clean Unused Models
Remove unused keras models in `saved_models`.

### Strip Vocabulary UNK token
Remove the `[UNK]` token from the vocabulary.txt file.

### Clean Training Checkpoints, Embeddings
See currently in-use checkpoint in `training/checkpoint`.
Update the `checkpoint` file to point to the
epoch of the best model saved in the archive.

### Update the `default_model/__init__.py`
Update the version and paths in the `__init__.py` to correspond to the
new model added.