# v1.12.0-rc1-10-g0a7de5f
Model trained with increased weights on benign Frq variants.

Model selection on epoch 34 where max(MCC) and also top highest score on `epoch_CommonVariantF1Frq`.

```
* 819d2c9 (vm-rdds1/master, vm-rdds0/master, origin/master, master) VRS: Minor gridlines and set axes for performance curves
| * aa5107f (HEAD -> 301/increase-frq-weights-default-model, tag: v1.12.0-rc-model, tag: merge-this) fixup! set model 301-common-frq-weights/20251011-043919-0a7de5f @ epoch 34 max MCC, CommonFrq
| * 24efb38 HACK: use model explainer bin from previous model
| * 55f6d71 set model 301-common-frq-weights/20251011-043919-0a7de5f @ epoch 34 max MCC, CommonFrq
| * 19badfd Model trained on 0a7de5f, 1600 weight
| * 0a7de5f (vm-rdds1/301/increase-frq-weights-default-model, origin/301/increase-frq-weights-default-model) Set common weight 1600
| * 67f903d Disable early stopping
| * 44bfddd HACK: Disable check numerics
| * dc18d2d VRS: Weights for local population benign variants
| * 15c26d2 VRS: Disable test dataset shuffle
| * 395927c vrs: Log test set TN/TP statistics
|/  
* 5de2979 Apptainer: Set tmpdir
* 61584b2 Makefile: Target to create mivmir singularity image
* 9f104e6 Add fn for adaptive learning rate
* dd04995 Git version: Rework function and improve container support
* 3e5587b (tag: v1.12.0-rc1) VRS: Add MIVMIR - Nextflow module to analyze model performance
```