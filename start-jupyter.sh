SIF_IMAGE_PATH=tmp/devenv/rdds_nvidia-dev-phen2gen.sif
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
	jupyter lab --config /rdds/build/devenv/jupyter_lab_config.py"

rmdir $SINGULARITY_CACHEDIR
rmdir $APPTAINER_TMPDIR
