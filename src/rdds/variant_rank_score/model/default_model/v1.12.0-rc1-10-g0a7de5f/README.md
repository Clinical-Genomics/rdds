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

## Case Performance Notes
The following notes relates to causative variants pulled from MUTACC database
and used by validation scripts to check performance.
Some cases have been ranked badly, and these are listed below for further
analysis.

### Cuteminnow. prio
Marked as VUS in scout, not reported by cust002.

### Finebedbug, prio
Annotated as 'criteria_provided&_conflicting_classifications' by Raredisease pipeline.
Newer information from ClinVar refer to this variant as pathogenic.
Rerunning with 'pathogenic&likely_pathogenic', 'criteria_provided&multiple_submitters&no_conflicts'
yields mivmir and gicam score 1.0.

### Romanticunicorn, prio
This variant was de novo, but dismissed because of lack of phenotype and low SpliceAI.
"Mark as causative" have been unchecked in cust002.

### Helpfulmarten
TBD

### Stirringtitmouse

Annotated 'criteria_provided&_conflicting_classifications' still in clinvar.
Expect no change in ranking performance.

## Thresholds
The following thresholds are equivalent GICAM thresholds to GENMOD (-30,+51 range):  

Genmod threshold 20 -> 0.618 normaliserat -> 0.85 recall -> 0.932 gicam threshold

genmod threshold 15 -> 0.556 normaliserat -> 0.911 recall -> 0.757 gicam threshold

genmod threshold 10 -> 0.494 normaliserat -> 0.927 recall -> 0.730 gicam threshold

