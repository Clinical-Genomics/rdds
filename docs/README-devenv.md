# Development Environment

The docker image `clinicalgenomics/rddsdev` is made available as a development environment.
The image contains all dependencies and various tools for machine learning
development.

The image can be pulled from `clinicalgenomics` organisation at dockerhub:
https://hub.docker.com/repository/docker/clinicalgenomics/rdds

`docker pull clinicalgenomics/rdds`

Features:
* Python virtualenv installation
* Graphics forwarding via X11 SSH

## Host Requirements
* `docker`, Docker-ce v20.10.21
* `singularity`, Singularity-ce v3.11.1
* `make` command, `dpkg;build-essential`
* `ssh`, openSSH

Please see [](README-devenv-prerequiesites.md) on how to
setup build tools on host machine.

## Docker and Singularity Support

## Build Process
The build process aims to create a singularity container that's runnable on
Hasta. In general, the process is to first build a docker image and then
convert it into singularity `.sif` format.

### Build the Docker Image
`make devenv-build`

### Build Singularity Image
Convert the devenv image to singularity format with the following command:

`make devenv-convert-dockerimage-to-singularity`

## Using the Development Image Locally
Build the docker image then start it by `make devenv-docker-sshd`.
Then SSH in to it by `make devenv-ssh`.

## Using the Development Image on Hasta (SLURM)
The main benefit using the devimage on Hasta is the data access.

### Image Installation
Install the `rdds[FLAVOR_VERSION].sif` file from singularity build step above, into
your repo on hasta, `[REPO_ROOT]/tmp/devenv/rdds[FLAVOR_VERSION].sif`.

### Starting Environment
1. Login to `hasta`
1. On hasta, start development environment by `make devenv-slurm-ubuntu_20_04-singularity-sshd`
2. Find what node the job runs on: `squeue -u $USER`
   To have the DNS name for this node, append `.local` to the node name, e.g `compute-0-6.local`.
3. Forward the SSH port to your local machine, eg `ssh -N -L 2150:compute-0-6.local:2150 hasta`
4. Login to container using `make devenv-ssh`

## Known Issues and Quirks

### Docker Mount UID-GID Limitation
This repository will be mounted into `/rdds` into the container,
and in docker root is the owner of the mount directory.
Files modified or created via the Docker mount point will
be owned by root on host machine.

It's better to work from Singularity in this case, since this issue is solved by
uid-guid mapping there.

### Remote Host Identification
When rebuilding the image, or pulling a new version the dropbear sshd key
will change. This will disable graphics forwarding on SSH client with the following
error:

```shell
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ECDSA key sent by the remote host is
SHA256:uxoEvFC+jL4J3WdguvDOP6YRWhV9C4yYMVrxhdKYxD4.
Please contact your system administrator.
Add correct host key in /home/USER/.ssh/known_hosts to get rid of this message.
Offending ECDSA key in /home/USER/.ssh/known_hosts:6
  remove with:
  ssh-keygen -f "/home/USER/.ssh/known_hosts" -R "[localhost]:2150"
Password authentication is disabled to avoid man-in-the-middle attacks.
Keyboard-interactive authentication is disabled to avoid man-in-the-middle attacks.
X11 forwarding is disabled to avoid man-in-the-middle attacks.
```
This will break X11 forwarding.
If this happens, just run the above command:
`ssh-keygen -f "/home/$USER/.ssh/known_hosts" -R "[localhost]:2150"`.

### GIT Colors on Hasta
Seems like Git does not provide colors by default
on Hasta. Run the following to force colored output:
`git config --global color.ui true`
