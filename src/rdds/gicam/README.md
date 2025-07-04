# GICAM
<h3>GENMOD Inheritance and Compound Adjustment for MIVMIR</h3>

This module merges inferences from [MIVMIR](../variant_rank_score) and
[GENMOD](https://github.com/Clinical-Genomics/genmod).

## TODOs
- [ ] Initial smoke test to validate that MIVMIR is better than GENMOD on 350+ cases

- [ ] Dataset containing real patient cases for comparing GICAM, GENMOD and MIVMIR and predictive power
  - [ ] Dataset split method to divide into train/ test sets (define a list of cases used for each set).

- [ ] Optimization routine for various module parameters
  - [ ] Tensorflow framework training script

- [ ] Determine the amount of samples required for statistical analysis
  - Determine the variance of inference improvement per case, this will impact amount of samples
    required. Is new module significantly better than default Genmod?

- [ ] Support for treating genmod configurable parameters as hyperparameters

Note that the datasets should support train/ test separation!
Need to have an index somehow on how to separate the variants in to sets.
Probably best to split on patient case name (sequencing time, patient).

Incorporating MIVMIR into Genmod scoring is mathematically equivalent
to summation of scores outside of Genmod, since Genmod is
linear in it's response.

## GENMOD Hyperparameters
Genmod has two [configurable parameters](https://github.com/Clinical-Genomics/genmod/blob/4799d3446c5476e56d74f9ebaf228f0a55f7048b/genmod/commands/score_compounds.py#L45-L51)
in the compound scoring step. `penalty` and `threshold` of int type.
- [ ] Support treating genmod config parameters as hyperparameters for tuning
  - [ ] Support for fast re-generation of GENMOD inferences in `nextflow-mivmir` repository

## Inference Merging Module
GENMOD is not a probabilistic module. Discrete steps in math.
Different step magnitude compared to MIVMIR.

Need to account for Genmod - MIVMIR differences in:
- bias, correct with `b`
- step size/ inference step resolution, correct with `w`

Do we need non-linearity in GENMOD function?
No, don't see how this would benefit (just adds another optimization parameter).

Options:
1. ~~`F = score_mivmir * score_genmod_normalized`~~
2. `F = score_mivmir * (w * score_genmod + b)` where `w, b` are tunable parameters
3. `F = score_mivmir * SIGMOID(w * score_genmod + b)`
4. `F = ((1 + b) * score_genmod_normalized * score_mivmir) / (b^2 * score_genmod_normalized * score_mivmir)`
    where `b` is a tunable parameter which sets importance of Genmod.
5. Weighted harmonic mean (4.) but with `score_genmod_corrected = w * score_genmod_normalized + b`

... something similar to Bayes theorem?
"Given a variant that follows inheritance pattern, what's the probabilty
it will be pathogenic?"

First option to try out is 2 and 5.

Optimization objective:
```
objective = MIN(Cost)
Cost = 1 - MCC_score
MCC_score = F(score_mivmir, score_genmod | variant_pathogenicity)
F(score_mivmir, score_genmod) = ((1 + b) * score_genmod_corrected * score_mivmir) / (b^2 * score_genmod_corrected * score_mivmir)
score_genmod_corrected = F(score_genmod_normalized, w, b) = w * score_genmod_normalized + b
```

## Software Structure
- [ ] Add new module for merging MIVMIR and Genmod? Yes, name it
  - MivmirWithInheritanceAndCompoundScores, MWICS
  - GenmodCorrectedMIVMIRScore, GCMS
  - InheritanceAndCompoundMivmirScore, ICMS
  - InheritanceCompoundAdjustmentModule, ICAM
  - GenmodInheritanceCompoundAdjustmentModule, GICAM
- Add method to rank variants in new module

## Power and Required Sample Size
Compute the power and estimated required sample size.

## Additional Metrics
### Number Needed to Treat, NNT
Nice alternative when comparing binary output models.

### Likelihood Ratio, LR
Comparison to expected genetic disease prevalence in patients.

## Readings
- [ ] [Evaluation of a decided sample size in machine learning applications](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05156-9)