Visualize hd5 dataset contents.

## Feature statistics

### Dataset and label skew
Should view every feature per
* class category, to view feature skew
* dataset category (train, test) to view skews

View every feature four times, per class and per dataset.

### Numerical features
- [x] Mean
- [x] stddev

### Text features
- [x] Word occurrence per feature
- [ ] Average sentence length per feature
- [ ] Word starting index in sentence

### Correlation Analysis
* [x] Visualize feature inter-correlation

### Nd Dimensionality Analysis
- [ ] PCA on numerical features
  - [ ] Solve issue with at least one NaN on every variant row (not allowed by `sklearn.decomposition.PCA`)
