#!/bin/bash
# Run model inference on VCF file.
# Expected runtime ~1h for 8.6E6 variants.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --time=7-00:00:00
#SBATCH --account=development
#SBATCH --job-name=RDDS-train
#SBATCH --qos=normal
#SBATCH --tmp=40G

set -e
set -x

TIMESTAMP=$(date +%s)
export SINGULARITY_CACHEDIR=`realpath tmp`/devenv/singularity-cache/$TIMESTAMP
mkdir -p $SINGULARITY_CACHEDIR

# Use --fakeroot to start as uid 0 and -w (required for sshd)
singularity exec \
--nv \
-w \
--fakeroot \
--no-home \
--cleanenv \
--contain \
--containall \
-B `pwd .`:/rdds \
tmp/devenv/rdds-v1.10.0-rc0.sif \
bash -c "\
export PYTHONPATH=/rdds/src && \
export PYTHONUNBUFFERED=1 &&
. /opt/pyenv/bin/activate && \
python3 -m rdds.variant_rank_score view-ranked-vcf \
/rdds/tmp/variant-rank-score/test_cases_explanations/*-predictions.vcf"

rmdir $SINGULARITY_CACHEDIR
