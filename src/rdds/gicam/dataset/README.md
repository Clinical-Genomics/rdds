# Dataset

Input files are original size about 1GB per case.
Totalling 350+ cases. About 56k variants per case.
Totalling ~19.6m variants, ~350 TPs.

## Dataset Use Cases
- Optimization
  - Create dataset and store as .csv
- Exploration
  - [ ] Visualisation of performance of GICAM compared to GENMOD default and standalone MIVMIR
  - Reuse existing visualizations in `vrs` module ? At least some portions can be ported to `lib/`.

## Dataset Structure

### Performance and Exploration Dataset
This dataset contains all annotations and all TN, TPs
(technically just a VcfReader instance).

Should all of variants be concatenated into a single VCF file or a .hd5 file?
No do this in RAM? Have lot's of RAM on Hasta.
However, parsing will be slow if we expect to rerun exploration
often. What about jupyter?

- [ ] TODO: Adjust Jupyter RAM limit to support very large datasets on Hasta

### Optimization Dataset
Load all variants TN, TP into a single dataset

```
|ID             |pathogenic |score_mivmir |score_genmod | case_name | set |
| 4.3123123 A>C | 1.0       |0.85         |0.42         | 0         | 0   |
```
this will be a small dataset that fits in RAM.

`class=1.0` is a pathogenic variant.
`set` is an integer mapping to `0: train, 1: test`.
Keep `case_name` as a lookup table, where `0: holyfish`.
`ID` is the variant ID, str type.

No need to add additional data features here, explorative performance analysis
is performed on VCF inference files.


## Dataset Generation