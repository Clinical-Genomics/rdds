# MIVMIR

This directory contains the
<b>M</b>odule for <b>I</b>ntegration of
<b>V</b>ariant <b>M</b>etadata for <b>I</b>nformed <b>R</b>anking (<b>MIVMIR</b>).

It's a regression type DNN model that infers pathogenicity score on SNVs,
trained on ClinVar TP and an Ashkenazim TN truth sets.

## Input Features
See the [model definition](model/model.py) for input, output specification.

Generally, the model is dependent on generally available upstream variant annotations,
_apart_ from the `Frq` annotation which is the variant frequency in the clinical, local cohort.
Suggest to use your own local database (if available) as this has a positive impact on performance.
The model can run without this annotation (input will then be set to `0.0`) but will result in
decreased performance.

## Training

Generally, the data sets are divided into
   *  ClinVar, negative background
   *  Validation set consisting of solved cases stored in our local MUTACC database

### Generating Datasets

1. Download ClinVar, GIAB and MUTACC datasets as VCF files.
   Refer to each of the `rdds.dataset_clinvar`, `rdds.dataset_giab` and `rdds/dataset_mutacc` for API usage.
   These modules will label and strip all additional information in VCF except
   `CHROM, POS, ID, REF, ALT, QUAL, FILTER and INFO/[LABEL]`.
3. Concat all VCF files using command `python3 -m rdds.variant_rank_score concat-vcfs [VCF_FILE, ...]`.
4. Run concatenated VCF file through `MIP` pipeline to annotate the variants as an case analysis.
5. Compile the `MIP` output VCF to `.hd5` using command `compile-vcf`.
6. Now you're ready to start training with the data using command `train`

The following VCF files are required for model training:
* [ ] Output from module `rdds.dataset_clinvar` containing True Positives, TPs (train data)
* [ ] Output from module `rdds.dataset_giab` containing True Negatives,  TNs (train data)
* [ ] Output from module `rdds.dataset_mutacc` containing True Positives, TPs (test, validation data)

### ClinVar
This is a dataset containing True Positive (pathogenic) variants.

```
python3 -m rdds.dataset_clinvar download-preprocess
```

### GIAB
This is a dataset containing True Negative (non-rare-disease causing) variants.
```
python3 -m rdds.dataset_giab download-preprocess
```

### MUTACC
This is a dataset containing clinically confirmed, rare-disease causing variants from
Clinical Genomics Stockholm.
```
src/rdds/dataset_mutacc/mutacc_formatter.sh \
tmp/mutacc-data/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf.gz \
tmp/mutacc-data/causative-variants.vcf.gz
```

Compilation to .hd5:
```
python3 -m \
rdds.variant_rank_score compile-vcf \
--vcf tmp/mutacc/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked-labeled.vcf.gz \
--dataset-file-name mutacc.hd5
```
Last known run configuration on Hasta required 5 cores and 120G of RAM.

### Vocabulary File
`models/vocabulary.txt` contains the embedding vocabulary.
This file should *not* contain `[UNK]` token, it's added
by Tensorflow embeddings layer when importing vocabulary.

However, for visualising the embeddings in Tensorboard,
please add the `[UNK]` token to the vocabulary text file.

## Hyperparameter Tuning
Enable the hyperparameter tuning flag when running training, example:
```
python3 -m rdds.variant_rank_score train /rdds/tmp/variant-rank-score/clinvar.hd5 --tune-hyperparams=1
```

## Run Inference on VCF
See the `predict-on-vcf` command.

Recommended CPU, RAM configuration is 10 cores and 150GB
of RAM.


## Automated Validation Suite (additional MUTACC data)
See the [mivmirvalidation](https://github.com/Clinical-Genomics/mivmirvalidation) repository.
This is a complete nextflow pipeline for:
* Pulling out solved, causative TPs from MUTACC alongside case background
* Running a case by case MIVMIR input feature annotation pipeline to generate MIVMIR input feature complete variants
* MIVMIR inference
* MIVMIR module validation performance scripts (and comparison to GENMOD)
* Generating additional training data for [GICAM module](../gicam)

### Manually Inspecting Model Performance
```bash
python3 -m rdds.variant_rank_score inference_exploration \
[KERAS_MODEL_PATH] \
[HD5_TRAIN_TEST_FILE_PATH]

cd tmp/variant-rank-score/inference_viz/
```
