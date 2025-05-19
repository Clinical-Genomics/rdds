.PHONY: *

DOCKERHUB=clinicalgenomics

DEFAULT_DEVENV_OS_FLAVOUR=ubuntu_20_04

DOCKER := DOCKER_BUILDKIT=1 docker
SINGULARITY_CACHEDIR_DEVENV=$(PWD)/tmp/devenv/singularity-cache-dir

# Any changes to versioning method, please update rdds.lib.git as well
VERSION=$(shell git describe --tags --dirty --always)
VERSION_LATEST_MASTER=$(shell git describe --tags --abbrev=0 origin/master)

all:
	# Default is no action
	@exit 0

get-version:
	@echo $(VERSION)

get-version-latest-master:
	 @echo $(VERSION_LATEST_MASTER)

devenv-nvidia-cudnn-build:
	# Image containing Nvidia cuDNN library
	$(DOCKER) build -t $(DOCKERHUB)/rdds-nvidia-cudnn:$(VERSION) --force-rm=true --rm=true -f build/devenv/nvidia-cudnn.Dockerfile .

devenv-nvidia-cudnn-push:
	$(DOCKER) push $(DOCKERHUB)/rdds-nvidia-cudnn:$(VERSION)

base-ubuntu-20-04-nvidia-build:
	# Build NVIDIA enabled docker image
	$(DOCKER) build -t $(DOCKERHUB)/rdds-ubuntu-20.04-nvidia:$(VERSION) --force-rm=true --rm=true - < build/devenv/ubuntu-20.04-nvidia.Dockerfile

base-ubuntu-20-04-nvidia-push:
	$(DOCKER) push $(DOCKERHUB)/rdds-ubuntu-20.04-nvidia:$(VERSION)

devenv-%-build:
	# Build docker development image
	# Valid targets are:
	# devenv-ubuntu_20_04-build (default)
	# devenv-ubuntu_20_04_nvidia-build (GPU enabled)
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) build \
	--build-arg="OS_FLAVOUR=$*" \
	--build-arg="VERSION=$(VERSION)" \
	-t $(DOCKERHUB)/rdds$(DEVENV_IMAGE_SUFFIX):$(VERSION) \
	--force-rm=true \
	--rm=true \
	--target devenv \
	-f build/devenv/devenv.Dockerfile .

vrs-production-image-build:
	# Build docker VRS model production inference image
	# This target is identical to devenv-%-build except for the --target.
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) build \
	--build-arg="OS_FLAVOUR=ubuntu_20_04" \
	--build-arg="VERSION=$(VERSION)" \
	-t $(DOCKERHUB)/rdds$(DEVENV_IMAGE_SUFFIX)_vrs:$(VERSION) \
	--force-rm=true \
	--rm=true \
	--target vrs-production \
	-f build/devenv/devenv.Dockerfile .

cosmograph-build:
	$(DOCKER) build \
	--build-arg="VERSION=$(VERSION)" \
	-t $(DOCKERHUB)/rdds_cosmograph:$(VERSION) \
	--force-rm=true \
	--rm=true \
	-f build/devenv/cosmograph.Dockerfile .

cosmograph-build-singularity-container:
	mkdir -p tmp/devenv
	docker save $(DOCKERHUB)/rdds_cosmograph:$(VERSION) -o tmp/devenv/rdds_cosmograph-$(VERSION).tar
	singularity build -F tmp/devenv/rdds_cosmograph-$(VERSION).sif docker-archive://tmp/devenv/rdds_cosmograph-$(VERSION).tar
	rm -f tmp/devenv/rdds_cosmograph-$(VERSION).tar

cosmograph-run:
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	mkdir -p $(SINGULARITY_CACHEDIR_DEVENV)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR_DEVENV) singularity exec --nv -w \
	--fakeroot --no-home --cleanenv --contain --containall \
	-B $(PWD):/rdds tmp/devenv/rdds_cosmograph-$(VERSION).sif /entrypoint.sh

    # 1. Install jupyterlab pip package in the environment you with to use
    # 2. Source the environment
	# 3. In container run: PYTHONPATH=/rdds/src jupyter lab --allow-root --no-browser

devenv-%-push:
	# Upload docker image to clinicalgenomics organisation at dockerhub
	# https://hub.docker.com/repository/docker/clinicalgenomics/rdds
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) push $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION)

vrs-production-image-push:
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) push $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}_vrs:$(VERSION)

devenv-%-docker-sshd:
	# Start development environment locally
	# Prefer the singularity option instead, since Docker docker mount
	# is always root:root owned.
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) run -it -l $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION) --rm=true -v $(PWD):/rdds -p 2150:2150 $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION)

devenv-ssh:
	# SSH into development container (local or tunneled)
	chmod go-rw build/devenv/devenv-docker.rsakey
	ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null devenv

devenv-ssh-hasta-%:
	# Connect to devenv at hasta node %, example; devenv-ssh-hasta-compute-0-7
	ssh -L 2151:$*:2150 -N -n hasta & \
	sleep 2 && \
	ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null -p 2151 devenv

devenv-%-convert-dockerimage-to-singularity:
	# Convert docker image to singularity format
	# Example: make devenv-ubuntu_20_04-convert-dockerimage-to-singularity
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	mkdir -p tmp/devenv
	docker save $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION) -o tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar
	singularity build -F tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).sif docker-archive://tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar
	rm -f tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar

devenv-local-%-singularity-sshd:
	# Start singularity development image locally
	# Use --fakeroot to start as uid 0 and -w (required for sshd)
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	mkdir -p $(SINGULARITY_CACHEDIR_DEVENV)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR_DEVENV) singularity exec --nv -w \
	--fakeroot --no-home --cleanenv --contain --containall \
	-B $(PWD):/rdds tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).sif /entrypoint.sh

devenv-slurm-%-singularity-sshd:
	# Start singularity development on SLURM
	# Example: make devenv-slurm-ubuntu_20_04-singularity-sshd
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	SIF_IMAGE_PATH=tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-${VERSION}.sif sbatch job.slurm /entrypoint.sh

docker-clean-images:
	# Remove all docker dangling images and stopped containers
	docker system prune

docker-clean-build-cache:
	# Removes docker build cache (often very large)
	docker builder prune
	docker buildx prune

singularity-clean-cache:
	# Remove cached images, builds in Singularity
	singularity cache clean

test:
	# Run tests on current version of docker image
	$(MAKE) test-$(VERSION)

test-master:
	# Run tests on latest master image
	docker pull clinicalgenomics/rdds:$(VERSION_LATEST_MASTER)
	$(MAKE) test-$(VERSION_LATEST_MASTER)

test-%:
	# Target for executing tests on various docker image versions (default flavour)
	$(DOCKER) run \
	-i \
	-l rdds-test:$* \
	--rm=true \
	-v $(PWD):/rdds \
	--entrypoint 'bash' \
	$(DOCKERHUB)/rdds:$* \
	-c \
	"export PYTHONPATH=/rdds/src && \
	. /opt/pyenv/bin/activate && \
	cd /rdds/src/tests && \
	python3 -m pytest -v -x lib && \
	python3 -m pytest -v -x exploration_rankscore && \
	python3 -m pytest -v -x variant_rank_score"

test-vrs-inference-cli:
	$(DOCKER) run \
	-it \
	--rm \
	-v ./src/tests/variant_rank_score:/data \
	$(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}_vrs:$(VERSION) /data/test_data.vcf

generate-dataset-statistics-%:
	# Run dataset statistics module to visualize dataset.
	# Expects a file in /rdds/tmp/clinvar.hd5
	# Example: `make generate-dataset-statistics-v1.2.0`
	$(DOCKER) run \
	-i \
	-l rdds-dataset-statistics:$* \
	--rm=true \
	-v $(PWD):/rdds \
	--entrypoint 'bash' \
	$(DOCKERHUB)/rdds:$* \
	-c \
	"export PYTHONPATH=/rdds/src && \
	. /opt/pyenv/bin/activate && \
	cd /rdds && \
	python3 -m rdds.lib.data_exploration tmp/clinvar.hd5"
