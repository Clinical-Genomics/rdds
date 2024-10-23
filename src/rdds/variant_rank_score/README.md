# Variant Rank Score Model

Previous genmod model used about ~50 parameters for estimating pathogenicity.

## TODOs
* [ ] Reduce parameters to about same size as genmod model
* [ ] Use the property of NaN values as a property of unseen, uncharacterized variants

## Features
The following features are used in genmod ranking model:
```
CSQ
	MaxEntScan_alt
	MaxEntScan_diff
	MES-SWA_acceptor_alt
	MES-SWA_acceptor_diff
	MES-SWA_donor_alt
	MES-SWA_donor_diff
	SpliceAI_pred_DS_AL
	SpliceAI_pred_DS_DG
	SpliceAI_pred_DS_DL
	PolyPhen
	REVEL_score
	SIFT
	LoFtool
	GERP++_RS
	phastCons100way_vertebrate
	phyloP100way_vertebrate
CLINVAR_CLNREVSTAT
CLINVAR_CLNSIG
CADD
FILTER
most_severe_consequence
ModelScore
SWEGENAF
GNOMADAF_popmax
SPIDEX
SpliceAI_pred_DS_AG
MTAF
Frq
GeneticModels
```

## Vocabulary File
`models/vocabulary.txt` contains the embedding vocabulary.
This file should *not* contain `[UNK]` token, it's added
by Tensorflow embeddings layer when importing vocabulary.

However, for visualising the embeddings in Tensorboard,
please add the `[UNK]` token.

## Generating Datasets

1. Download ClinVar, GIAB and MUTACC datasets as VCF files.
   Refer to each of the `rdds.dataset_clinvar`, `rdds.dataset_giab` and `rdds/dataset_mutacc` for API usage.
   These modules will label and strip all additional information in VCF except
   `CHROM, POS, ID, REF, ALT, QUAL, FILTER and INFO/[LABEL]`.
3. Concat all VCF files using command `python3 -m rdds.variant_rank_score concat-vcfs [VCF_FILE, ...]`.
4. Run concatenated VCF file through `MIP` pipeline to annotate the variants.
5. Compile the `MIP` output VCF to `.hd5` using command `TBD`.
6. Now you're ready to start training with the data using command `TBD`

The following VCF files are required for model training:
* [ ] Output from module `rdds.dataset_clinvar` containing True Positives, TPs (train data)
* [ ] Output from module `rdds.dataset_giab` containing True Negatives,  TNs (train data)
* [ ] Output from module `rdds.dataset_mutacc` containing True Positives, TPs (test, validation data)

### ClinVar
This is a dataset containing True Positive (pathogenic) variants.

```
python3 -m rdds.dataset_clinvar preprocess
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

## Hyperparameter Tuning
Enable the hyperparamter tuning flag when running training, example:
```
python3 -m rdds.variant_rank_score train /rdds/tmp/variant-rank-score/clinvar.hd5 --tune-hyperparams=1
```
