# Variant Rank Score Model

Previous genmod model used about ~50 parameters for estimating pathogenicity.

## TODOs
* [ ] Reduce parameters to about same size as genmod model


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
