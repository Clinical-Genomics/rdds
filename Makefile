.PHONY: *

DOCKER := DOCKER_BUILDKIT=1 docker

all:
	# Default is no action
	@exit 0

devenv-build:
	# Build docker development image
	$(DOCKER) build -t cg/rdds --force-rm=true --rm=true --target devenv -f build/devenv/devenv.Dockerfile .

devenv-docker-sshd:
	# Start development environment locally
	# Prefer the singularity option instead, since Docker docker mount
	# is always root:root owned.
	$(DOCKER) run -it -l cg/rdds --rm=true -v $(PWD):/rdds -p 2150:2150 cg/rdds

devenv-ssh:
	# SSH into development container (local or tunneled)
	ssh -F build/devenv/devenv.sshconfig -oUserKnownHostsFile=/dev/null devenv

devenv-convert-dockerimage-to-singularity:
	# Convert docker image to singularity format
	mkdir -p tmp/devenv
	docker save cg/rdds -o tmp/devenv/rdds.tar
	singularity build -F tmp/devenv/rdds.sif docker-archive://tmp/devenv/rdds.tar
	rm -f tmp/devenv/rdds.tar

devenv-singularity-sshd:
	# Start singularity development image locally
	# Use --fakeroot to start as uid 0 and -w (required for sshd)
	singularity exec -w --fakeroot --no-home --cleanenv --contain --containall -B $(PWD):/rdds tmp/devenv/rdds.sif /entrypoint.sh

docker-clean-images:
	# Remove all docker dangling images and stopped containers
	docker system prune
