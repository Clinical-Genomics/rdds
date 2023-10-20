.PHONY: *

DOCKERHUB=clinicalgenomics

DEFAULT_DEVENV_OS_FLAVOUR=ubuntu_20_04

DOCKER := DOCKER_BUILDKIT=1 docker
SINGULARITY_CACHEDIR_DEVENV=$(PWD)/tmp/devenv/singularity-cache-dir

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

devenv-%-build:
	# Build docker development image
	# Valid targets are:
	# devenv-ubuntu_20_04-build (default)
	# devenv-ubuntu_20_04_nvidia_470-build (GPU enabled)
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) build \
	--build-arg="OS_FLAVOUR=$*" \
	--build-arg="VERSION=$(VERSION)" \
	-t $(DOCKERHUB)/rdds$(DEVENV_IMAGE_SUFFIX):$(VERSION) \
	--force-rm=true \
	--rm=true \
	--target devenv \
	-f build/devenv/devenv.Dockerfile .

devenv-%-push:
	# Upload docker image to clinicalgenomics organisation at dockerhub
	# https://hub.docker.com/repository/docker/clinicalgenomics/rdds
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	$(DOCKER) push $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION)

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

devenv-%-convert-dockerimage-to-singularity:
	# Convert docker image to singularity format
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	mkdir -p tmp/devenv
	docker save $(DOCKERHUB)/rdds${DEVENV_IMAGE_SUFFIX}:$(VERSION) -o tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar
	singularity build -F tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).sif docker-archive://tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar
	rm -f tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).tar

devenv-%-singularity-sshd:
	# Start singularity development image locally
	# Use --fakeroot to start as uid 0 and -w (required for sshd)
	$(eval DEVENV_IMAGE_SUFFIX=$(subst $(DEFAULT_DEVENV_OS_FLAVOUR),,$*))
	mkdir -p $(SINGULARITY_CACHEDIR_DEVENV)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR_DEVENV) singularity exec --nv -w \
	--fakeroot --no-home --cleanenv --contain --containall \
	-B $(PWD):/rdds tmp/devenv/rdds${DEVENV_IMAGE_SUFFIX}-$(VERSION).sif /entrypoint.sh

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
	. /opt/conda/bin/activate && \
	cd /rdds/src/tests && python3 -m pytest -v"
