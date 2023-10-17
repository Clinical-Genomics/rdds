.PHONY: *

DOCKERHUB=clinicalgenomics
DOCKER := DOCKER_BUILDKIT=1 docker
SINGULARITY_CACHEDIR_DEVENV=$(PWD)/tmp/devenv/singularity-cache-dir

TAG=$(shell git describe --tags --dirty --always)
TAG=gpu

all:
	# Default is no action
	@exit 0

devenv-build:
	# Build docker development image
	#$(DOCKER) build -t $(DOCKERHUB)/rdds:$(TAG) --force-rm=true --rm=true --target devenv -f build/devenv/devenv.Dockerfile .
	DOCKER_BUILDKIT=0 docker build -t $(DOCKERHUB)/rdds:$(TAG) --force-rm=true --rm=true --target devenv - < build/devenv/devenv.Dockerfile

devenv-upload-clinicalgenomics-dockerhub:
	# Upload docker image to clinicalgenomics organisation at dockerhub
	# https://hub.docker.com/repository/docker/clinicalgenomics/rdds
	$(DOCKER) push $(DOCKERHUB)/rdds:$(TAG)

devenv-docker-sshd:
	# Start development environment locally
	# Prefer the singularity option instead, since Docker docker mount
	# is always root:root owned.
	$(DOCKER) run -it -l $(DOCKERHUB)/rdds:$(TAG) --rm=true -v $(PWD):/rdds -p 2150:2150 $(DOCKERHUB)/rdds:$(TAG)

devenv-ssh:
	# SSH into development container (local or tunneled)
	chmod go-rw build/devenv/devenv-docker.rsakey
	ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null devenv

devenv-convert-dockerimage-to-singularity:
	# Convert docker image to singularity format
	mkdir -p tmp/devenv
	docker save $(DOCKERHUB)/rdds:$(TAG) -o tmp/devenv/rdds-$(TAG).tar
	singularity build -F tmp/devenv/rdds-$(TAG).sif docker-archive://tmp/devenv/rdds-$(TAG).tar
	rm -f tmp/devenv/rdds-$(TAG).tar

devenv-singularity-sshd:
	# Start singularity development image locally
	# Use --fakeroot to start as uid 0 and -w (required for sshd)
	mkdir -p $(SINGULARITY_CACHEDIR_DEVENV)
	SINGULARITY_CACHEDIR=$(SINGULARITY_CACHEDIR_DEVENV) singularity exec --nv -w --fakeroot --no-home --cleanenv --contain --containall -B $(PWD):/rdds tmp/devenv/rdds-$(TAG).sif /entrypoint.sh

docker-clean-images:
	# Remove all docker dangling images and stopped containers
	docker system prune
