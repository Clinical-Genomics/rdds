#!/bin/bash
# Tested with:
#   - bcftools 1.10.2 (using htslib 1.10.2-3ubuntu0.1)
#   - tabix 1.10.2-3ubuntu0.1
# Script to add field INFO/GIAB_GROUND_TRUTH=[benign] to
# GIAB VCF data files for data exploration.
# $1: GIAB data file, bgzipped
set -e
set -x
export DATAFILE=`realpath $1`
stat $DATAFILE &>/dev/null
export OUTFILE=`echo $DATAFILE | sed 's/\.vcf\.gz//g'`-labeled.vcf.gz

function reindex_vcf() {
  # Create VCF index to allow processing by bcftools
  # $1: Path to VCF to reindex
  tabix -f -p vcf $1
}

function add_label() {
  # Label variants in VCF with INFO/GIAB_GROUND_TRUTH=[str:$2] used as ground truth label
  # $1: Path to VCF to label
  # $2: String to add to INFO/GIAB_GROUND_TRUTH field
  local OUTFILE=`echo $1 | sed 's/\.vcf\.gz//g'`-labeled.vcf
  ## Add INFO/GIAB_GROUND_TRUTH definition to header
  bcftools view --header $1 | head -n-1 > $OUTFILE
  echo "##INFO=<ID=GIAB_GROUND_TRUTH,Number=.,Type=String,Description=\"Genome In A Bottle - ClinicalGenomics inferred label\">" >> $OUTFILE
  # Set up custom header required by MIP pipeline
  echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNOTASAMPLE" >> $OUTFILE
  ## Add variant data with INFO/GIAB_GROUND_TRUTH appended
  # Thus these fields are incompatible with bcftools and are consequently dropped.
  bcftools query -f "%CHROM\t%POS\t%ID\t%REF\t%ALT\t.\t.\tGIAB_GROUND_TRUTH=$2\tGT:DP:AD:GQ\t1/1:30:4,26:38\n" $1 >> $OUTFILE
  # Compress it to allow downstream processing by bcftools
  bgzip --threads 4 $OUTFILE
  reindex_vcf $OUTFILE.gz
}

function check_no_missing_samples() {
  # Check that no variants are missing in final dataset, list the missing variants
  # $1: Reference VCF
  # $2: (possibly modified) VCF
  bcftools isec -p dset-diff -n-1 -c all $1 $2
  # Test for empty sites.txt file (no difference observed)
  if (( `stat -c%s dset-diff/sites.txt` != 0 ))
  then
    echo "Not all samples transferred to labeled dset file:"
    cat dset-diff/sites.txt
    false
  fi
}

# Add label to variants
add_label $DATAFILE benign

# Check no data loss
check_no_missing_samples $DATAFILE $OUTFILE

rm -r dset-diff

echo Output file: $OUTFILE
