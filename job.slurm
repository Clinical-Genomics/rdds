#!/bin/bash
# Execute arbitrary command in RDDS container on SLURM
# Example: SIF_IMAGE_PATH=tmp/devenv/rdds-v1.4.0.sif sbatch job.slurm "python3 -c \"print('helloworld')\""
# Requires that SIF_IMAGE_PATH is set in environment
# Expected to be run from repo root.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=90G
#SBATCH --time=1-00:00:00
#SBATCH --account=development
#SBATCH --job-name=RDDS-dev
#SBATCH --qos=normal
#SBATCH --tmp=20G

set -e
set -x

if [ "$SIF_IMAGE_PATH" == '' ];
then
	echo "Expected SIF_IMAGE_PATH to be set" > /dev/stderr
        exit 1
fi

TIMESTAMP=$(date +%s)
export SINGULARITY_CACHEDIR=`realpath tmp`/devenv/singularity-cache/$TIMESTAMP
export APPTAINER_TMPDIR=`realpath tmp`/devenv/apptainer-cache/$TIMESTAMP
mkdir -p $SINGULARITY_CACHEDIR
mkdir -p $APPTAINER_TMPDIR

# Use --fakeroot to start as uid 0 and -w (required for sshd)
singularity exec \
`./apptainer-args.sh` \
--nv \
-w \
--fakeroot \
--no-home \
--cleanenv \
--contain \
--containall \
-B `pwd .`:/rdds \
$(realpath $SIF_IMAGE_PATH) \
bash -c "\
export PYTHONPATH=/rdds/src && \
export PYTHONUNBUFFERED=1 &&
. /opt/pyenv/bin/activate && \
$@ "

rmdir $SINGULARITY_CACHEDIR
rmdir $APPTAINER_TMPDIR
