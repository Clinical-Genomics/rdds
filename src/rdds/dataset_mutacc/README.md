# Clinical Genomics Stockholm MUTACC Database Module

This module reads a MUTACC database VCF excerpt
`mutacc-[YYYYMMDD]_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf.gz` containing
true positive (TP) disease causing and true negative (TN) variants and a
list of TPs `causative-variants.vcf.gz` to generate a merged, labeled VCF file.

Example usage:
```
./mutacc_formatter.sh \
mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf.gz \
causative-variants.vcf.gz
[...]
Output file: /rdds/tmp/mutacc/wdir/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked-labeled.vcf.gz
```
