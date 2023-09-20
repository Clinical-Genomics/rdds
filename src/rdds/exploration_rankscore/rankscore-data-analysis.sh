#!/bin/bash

# TODO: Add time to commands

set -e

. /opt/conda/bin/activate

export PYTHONPATH=/rdds/src
export DDIR=/rdds/tmp/exploration-rankscore/data
export LOGDIR=/rdds/tmp/exploration-rankscore
export PYTHONUNBUFFERED=1
# NOTE to self: Done use genmod score here, reference genmod file is from _compound step.
export VCF=$DDIR/normalized_pr/snv_indel/genmod_compound/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed.annotate_models_score_compound.vcf
export VCF_SV=$DDIR/normalized_pr/sv/genmod_compound/mutacc-20230512_comb_ann_vep_parsed.annotate_models_score_compound.vcf
export VCF_REF=$DDIR/genmod_v3.7.3/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf
export MUTACC_TRUTH=$DDIR/mutacc-20230512_truth/causative-variants.vcf

# Compile datasets
python3 -m rdds.exploration_rankscore compile --vcf $VCF --dataset-file-name `basename $VCF`-snv.hd5 --features RankScore,RankScoreNormalized,RankScoreMinMax --vcf-mutacc-tp-cases $MUTACC_TRUTH >$LOGDIR/dset.log 2>&1 &
python3 -m rdds.exploration_rankscore compile --vcf $VCF_SV --dataset-file-name `basename $VCF_SV`-sv.hd5 --features RankScore,RankScoreNormalized,RankScoreMinMax --vcf-mutacc-tp-cases $MUTACC_TRUTH >$LOGDIR/dset_sv.log 2>&1 &
python3 -m rdds.exploration_rankscore compile --vcf $VCF_REF --dataset-file-name `basename $VCF_REF`-ref.hd5 --features RankScore >$LOGDIR/dsetref.log 2>&1 &

disown -a
exit 0

# Run analysis
export HD5_SNV=$LOGDIR/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed.annotate_models_score_compound.vcf-snv.hd5
export HD5_SV=$LOGDIR/mutacc-20230512_comb_ann_vep_parsed.annotate_models_score_compound.vcf-sv.hd5
export HD5_REF=$LOGDIR/mutacc-20230512_gatkcomb_rhocall_norm_af_mt_frqf_cadd_vep_parsed_ranked.vcf-ref.hd5

python3 -m rdds.exploration_rankscore testnormalizedrankscore --hd5 $HD5_SNV --hd5ref $HD5_REF > $LOGDIR/testnormalizedrankscore-snv.log 2>&1 &
python3 -m rdds.exploration_rankscore testnormalizedrankscore --hd5 $HD5_SV --hd5ref $HD5_REF > $LOGDIR/testnormalizedrankscore-sv.log 2>&1 &

python3 -m rdds.exploration_rankscore rankscorestats --hd5 $HD5_SNV --image-name-prefix snv > $LOGDIR/rankscorestats-snv.log 2>&1 &
python3 -m rdds.exploration_rankscore rankscorestats --hd5 $HD5_SV --image-name-prefix sv --k-fold-subset-size 25000 > $LOGDIR/rankscorestats-sv.log 2>&1 &

disown -a
exit 0