# v1.11.0-rc3-3-gb427979

* Removes SPIDEX annotation, fb6aa4d
* Bugfix related to Embeddings reduction, d4d3833

```
* 88e1261 (HEAD -> 231/remove-spidex-annotation, origin/231/remove-spidex-annotation, hasta/231/remove-spidex-annotation) HACK: VRS: DefModel w non-spidex normalisation, vocabulary no token
* fb6aa4d VRS: Drop SPIDEX feature
* 33cb65c VRS: Dynamic idx for rare variant weight input frq tensors
* 37a2bd7 VRS: Remove commented out input feature MTAF
* 3decc72 (origin/next, next) EmbeddingsReductionLayer: Move inline comment
* fdead37 test embeddings: Adjustments for reduce_sum reduction
* d4d3833 EmbeddingsReductionLayer: Bugfix for embedding leakage across batchDim
* 32ba076 test embeddings: bugfix; compute sum of embeddings with abs()
* 5b2f10b test embeddings: Adjust test data to capture embeddings reduction inconsistency
* 1c5c33b VRS test: Add method to compare layer nd output arrays
* b16cd39 test embeddings: Add test for batch agnostic embeddings generation
* d4be5eb VRS inference test: Variant integrity in single vs batch mode
* b7c9ae9 VRS inference test: Parametrize CPU count
* 676be28 VRS: infer score: Remove invalid inline comment
* 9bee996 VRS: Update test data, generated from single core
* ea01201 Revert HACK: Disable VRS test, issue 201
* 6ced50b VRS: Use VCF reference file for inference testing
* ec9bdf1 test: Move fixture work_dir, accessible to all tests
* 3bf6dfe (tag: v1.11.0, origin/master) VRS prod docker image: Drop entrypoint, rename images to MIVMIR
```
