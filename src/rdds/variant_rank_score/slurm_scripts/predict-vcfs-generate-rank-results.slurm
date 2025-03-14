#!/bin/bash
# Run model inference on VCF file(s) and compute performance comparison with Genmod.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --time=1-00:00:00
#SBATCH --account=development
#SBATCH --job-name=RDDS-vcfinf
#SBATCH --qos=normal
#SBATCH --tmp=40G

set -e
set -x

SINGULARITY_IMAGE=tmp/devenv/rdds-v1.10.0-rc0.sif
CASES_DIR=tmp/variant-rank-score/test_cases_next
CPU_CORES=10

TIMESTAMP=$(date +%s)
export SINGULARITY_CACHEDIR=`realpath tmp`/devenv/singularity-cache/$TIMESTAMP
mkdir -p $SINGULARITY_CACHEDIR

# Generate predictions
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
$SINGULARITY_IMAGE \
bash -c "\
export PYTHONPATH=/rdds/src && \
export PYTHONUNBUFFERED=1 &&
. /opt/pyenv/bin/activate && \
python3 -m rdds.variant_rank_score predict-on-vcf \
/rdds/$CASES_DIR/*ranked.vcf \
--pretrained_model_path \
/rdds/8508060_dropout_05.keras \
--pretrained_model_explainer_path \
/rdds/src/rdds/variant_rank_score/model/default_model/v1.11.0/model-explainer.bin \
--cpu_cores $CPU_CORES"

# Analyze predictions
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
$SINGULARITY_IMAGE \
bash -c "\
export PYTHONPATH=/rdds/src && \
export PYTHONUNBUFFERED=1 &&
. /opt/pyenv/bin/activate && \
python3 -m rdds.variant_rank_score view-ranked-vcf \
/rdds/$CASES_DIR/*-predictions.vcf"

rmdir $SINGULARITY_CACHEDIR
