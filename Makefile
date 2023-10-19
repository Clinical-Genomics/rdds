.PHONY: *

DOCKERHUB=clinicalgenomics

# Choose from CPU or GPU enabled Tensorflow development
DEFAULT_DEVENV_OS_FLAVOUR=ubuntu_20_04
UBUNTU_GPU_ENABLED_FLAVOUR=ubuntu_20_04_nvidia_470
DEVENV_OS_FLAVOUR=${DEFAULT_DEVENV_OS_FLAVOUR}
DEVENV_IMAGE_SUFFIX=$(shell echo ${DEVENV_OS_FLAVOUR} | sed -e 's/${DEFAULT_DEVENV_OS_FLAVOUR}//g')

DOCKER := DOCKER_BUILDKIT=1 docker
SINGULARITY_CACHEDIR_DEVENV=$(PWD)/tmp/devenv/singularity-cache-dir

TAG=$(shell git describe --tags --dirty --always)
TAG=gpu

all:
	# Default is no action
	@exit 0

devenv-nvidia-cudnn-build:
	# Image containing Nvidia cuDNN library
	$(DOCKER) build -t $(DOCKERHUB)/rdds-nvidia-cudnn:$(TAG) --force-rm=true --rm=true -f build/devenv/nvidia-cudnn.Dockerfile .

devenv-nvidia-cudnn-push:
	$(DOCKER) push $(DOCKERHUB)/rdds-nvidia-cudnn:$(TAG)

devenv-ubuntu-20.04-nvidia-470-build:
	# Image containing Ubuntu 20.04 with NVIDIA GPU driver 470.
	$(DOCKER) build -t $(DOCKERHUB)/rdds-ubuntu-20.04-nvidia-470:$(TAG) --force-rm=true --rm=true -f build/devenv/ubuntu-20.04-nvidia-470.Dockerfile .

devenv-ubuntu-20.04-nvidia-470-push:
	$(DOCKER) push $(DOCKERHUB)/rdds-ubuntu-20.04-nvidia-470:$(TAG)

devenv-build:
	# Build docker development image
	$(DOCKER) build --build-arg="OS_FLAVOUR=${DEVENV_OS_FLAVOUR}" -t $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(TAG) --force-rm=true --rm=true --target devenv -f build/devenv/devenv.Dockerfile .

devenv-push:
	# Upload docker image to clinicalgenomics organisation at dockerhub
	# https://hub.docker.com/repository/docker/clinicalgenomics/rdds
	$(DOCKER) push $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(TAG)

devenv-docker-sshd:
	# Start development environment locally
	# Prefer the singularity option instead, since Docker docker mount
	# is always root:root owned.
	$(DOCKER) run -it -l $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(TAG) --rm=true -v $(PWD):/rdds -p 2150:2150 $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(TAG)

devenv-ssh:
	# SSH into development container (local or tunneled)
	chmod go-rw build/devenv/devenv-docker.rsakey
	ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null devenv

devenv-convert-dockerimage-to-singularity:
	# Convert docker image to singularity format
	mkdir -p tmp/devenv
	docker save $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(TAG) -o tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(TAG).tar
	singularity build -F tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(TAG).sif docker-archive://tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(TAG).tar
	rm -f tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(TAG).tar

devenv-singularity-sshd:
	# Start singularity development image locally
	# Use --fakeroot to start as uid 0 and -w (required for sshd)
	mkdir -p $(SINGULARITY_CACHEDIR_DEVENV)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR_DEVENV) singularity exec --nv -w --fakeroot --no-home --cleanenv --contain --containall -B $(PWD):/rdds tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(TAG).sif /entrypoint.sh

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
