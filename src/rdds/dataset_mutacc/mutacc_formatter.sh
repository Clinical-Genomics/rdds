#!/bin/bash
# Tested with:
#   - bcftools 1.10.2 (using htslib 1.10.2-3ubuntu0.1)
#   - tabix 1.10.2-3ubuntu0.1
# Script to add field INFO/MUTACC_GROUND_TRUTH=[benign|pathogenic] to
# MUTACC VCF data files for data exploration.
# Only tested with SNVs and INDELs (separate file for DEL dups and SVs).
# $1: MUTACC data file, bgzipped
# $2: causative variants data file, bgzipped
# $3: [--preserve-tns: do not strip negative samples from MUTACC file]
set -e
set -x
export DATAFILE_MUTACC=`realpath $1`
stat $DATAFILE_MUTACC &>/dev/null
export DATAFILE_CAUSATIVE=`realpath $2`
stat $DATAFILE_CAUSATIVE &>/dev/null
export OUTFILE=`echo $DATAFILE_MUTACC | sed 's/\.vcf.gz//g'`-labeled.vcf.gz

function reindex_vcf() {
  # Create VCF index to allow processing by bcftools
  # $1: Path to VCF to reindex
  tabix -f -p vcf $1
}

function add_label() {
  # Label variants in VCF with INFO/MUTACC_GROUND_TRUTH=[str:$2] used as ground truth label
  # $1: Path to VCF to label
  # $2: String to add to INFO/MUTACC_GROUND_TRUTH field
  local OUTFILE=`echo $1 | sed 's/\.vcf//g'`-labeled.vcf
  ## Add INFO/MUTACC_GROUND_TRUTH definition to header
  bcftools view --header $1 | head -n-1 > $OUTFILE
  echo "##INFO=<ID=MUTACC_GROUND_TRUTH,Number=.,Type=String,Description=\"Clinicalgenomics MUTACC label\">" >> $OUTFILE
  # Set up custom header required by MIP pipeline
  echo -e "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNOTASAMPLE" >> $OUTFILE
  ## Add variant data with INFO/MUTACC_GROUND_TRUTH appended
  bcftools query -f "%CHROM\t%POS\t%ID\t%REF\t%ALT\t%QUAL\t%FILTER\tMUTACC_GROUND_TRUTH=$2\tGT:DP:AD:GQ\t1/1:30:4,26:38\n" $1 >> $OUTFILE
  # Compress it to allow downstream processing by bcftools
  bgzip --threads 4 $OUTFILE
  reindex_vcf $OUTFILE.gz
}

function concat_tps_tns() {
  # Merge TPs and TNs variants to final dataset
  # $1: TPs
  # $2: TNs
  bcftools concat --allow-overlaps $1 $2 -o concat.vcf
  bgzip --threads 4 concat.vcf
  mv concat.vcf.gz $OUTFILE
  reindex_vcf $OUTFILE
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

# Reindex files
reindex_vcf $DATAFILE_MUTACC
reindex_vcf $DATAFILE_CAUSATIVE

# Select causative from mutacc data file using causative positions as reference
bcftools isec -p tps -n=2 -w1 $DATAFILE_MUTACC $DATAFILE_CAUSATIVE

# Select non-causative from mutacc data by selecting all but the causative
# positions in the causative file.
bcftools isec -p tns --complement -w1 $DATAFILE_MUTACC $DATAFILE_CAUSATIVE

# Add label to variants
add_label tps/0000.vcf pathogenic
add_label tns/0000.vcf benign

if [ "$3" == "--preserve-tns" ]
then
  # Merge tps, tns files
  concat_tps_tns tps/0000-labeled.vcf.gz tns/0000-labeled.vcf.gz

  # Check no data loss
  check_no_missing_samples $DATAFILE_MUTACC $OUTFILE

else
  mv tps/0000-labeled.vcf.gz $OUTFILE
  reindex_vcf $OUTFILE
fi

rm -rf tns \
  tps \
  dset-diff \
  $DATAFILE_MUTACC.tbi \
  $DATAFILE_CAUSATIVE.tbi

echo Output file: $OUTFILE
