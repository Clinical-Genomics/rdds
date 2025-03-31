#!/bin/bash
# Execute arbitrary command in RDDS container on SLURM
# Expected to be run from repo root.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=100G
#SBATCH --time=0-06:00:00
#SBATCH --account=development
#SBATCH --job-name=RDDS-normparam
#SBATCH --qos=low
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
tmp/devenv/rdds-v1.10.0.sif \
bash -c "\
export PYTHONPATH=/rdds/src && \
export PYTHONUNBUFFERED=1 &&
. /opt/pyenv/bin/activate && \
python3 -m rdds.variant_rank_score train \
/rdds/tmp/variant-rank-score/compiletest-vepfixes-final-compile-v1.5.0.hd5 \
--compile-vocabulary-normalisation-factors 1"

rmdir $SINGULARITY_CACHEDIR
