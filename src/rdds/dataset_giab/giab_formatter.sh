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
  bcftools view --header $1 | grep "\#\#fileformat=" > $OUTFILE
  bcftools view --header $1 | grep "\#\#contig=" >> $OUTFILE
  bcftools view --header $1 | grep "\#\#FORMAT=<ID=GT," >> $OUTFILE
  bcftools view --header $1 | grep "\#\#FORMAT=<ID=DP," >> $OUTFILE
  # Some multiallelic sites does not contain proper AD values in dataset.
  # Allow for missing values when splitting sites with 'bcftools norm' later on. https://github.com/samtools/bcftools/issues/823
  echo "##FORMAT=<ID=AD,Number=.,Type=Integer,Description=\"Net allele depths across all unfiltered datasets with called genotype\">" >> $OUTFILE
  bcftools view --header $1 | grep "\#\#FORMAT=<ID=GQ," >> $OUTFILE
  ## Add INFO/GIAB_GROUND_TRUTH definition to header
  echo "##INFO=<ID=GIAB_GROUND_TRUTH,Number=.,Type=String,Description=\"Genome In A Bottle - ClinicalGenomics inferred label\">" >> $OUTFILE
  # Set up custom header required by MIP pipeline
  echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNOTASAMPLE" >> $OUTFILE
  ## Add variant data with INFO/GIAB_GROUND_TRUTH appended
  # Thus these fields are incompatible with bcftools and are consequently dropped.
  bcftools query -f "%CHROM\t%POS\t%ID\t%REF\t%ALT\t.\t.\tGIAB_GROUND_TRUTH=$2\tGT:DP:AD:GQ\t[%GT:%DP:%AD:%GQ]\n" $1 >> $OUTFILE
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

function split_multiallelic_sites() {
  local OUTFILE=`echo $1 | sed 's/\.gz//g'`
  bcftools norm -m-both $1 > $OUTFILE
  bgzip --threads 4 $OUTFILE
  reindex_vcf $OUTFILE.gz
}

# Add label to variants
add_label $DATAFILE benign

# Check no data loss
check_no_missing_samples $DATAFILE $OUTFILE

split_multiallelic_sites $OUTFILE

rm -r dset-diff

echo Output file: $OUTFILE
