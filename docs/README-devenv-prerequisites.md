# Setting Up Local Development Environment

This scope covers installation of host tools on Ubuntu 20.04 LTS.

## Singularity
Singularity is used for containerisation of development and
production images.

### Install OS deps
https://docs.sylabs.io/guides/main/user-guide/quick_start.html
```
# Ensure repositories are up-to-date
sudo apt-get update
# Install debian packages for dependencies
sudo apt-get install -y \
   wget \
   build-essential \
   libseccomp-dev \
   libglib2.0-dev \
   pkg-config \
   squashfs-tools \
   cryptsetup \
   runc \
   git \
   libssl-dev \
   debootstrap
```

### Install Golang
Golang is the language of choice in the Singularity project.

https://go.dev/doc/install

**WARN**: This will remove any previously existing golang installation
on host machine!
```
(
export GOVER=go1.20.3.linux-amd64 && \
cd /tmp && \
wget https://go.dev/dl/$GOVER.tar.gz && \
sudo rm -rf /usr/local/go && \
sudo mkdir /usr/local/go && \
sudo chown $USER:$USER /usr/local/go && \
tar --one-top-level=/usr/local -xzf $GOVER.tar.gz
)
```
Make golang binaries accessible to user
```
echo "export PATH=$PATH:/usr/local/go/bin" >> ~/.bashrc && \
source ~/.bashrc
```

Setup go dir for user source code:
```
sudo mkdir -p /usr/local/go-path && \
sudo chown $USER:$USER /usr/local/go-path && \
echo "export GOPATH=/usr/local/go-path" >> ~/.bashrc && \
source ~/.bashrc
```
Adjust go [mod policy](https://go.dev/ref/mod#mod-commands)
(disable module mode, use $GOPATH)
```
go env -w GO111MODULE=off
```

### Build and Install Singularity
This guide installs v3.11.1,
compared to HASTA singularity version v3.1.1.
The reason for upgrading this locally is that the container library
endpoints have changed (breaking change) rendering v3.1.1
version unusable for building new container images using.
This is because library endpoint cloud.sylabs.io/library is not
properly functioning in v3.1.1 due to bad formatted URI and query
strings. The old library endpoint library.sylabs.io is no longer
accessible.

**WARN**:  This will remove any already existing repo `singularity`.
```
(
export SINGULARITYVERSION=v3.11.1 && \
export WDIR=$GOPATH/src/github.com/sylabs && \
mkdir -p $WDIR && \
cd $WDIR && \
rm -rf singularity && \
git clone --recurse-submodules https://github.com/sylabs/singularity.git && \
cd singularity && \
git checkout $SINGULARITYVERSION && \
./mconfig && \
    make -j 4 -C builddir && \
    sudo make -j 4 -C builddir install
)
```
Test Singularity install
```
singularity version
```

## Known Issues
### Loopback devices issue
```
> singularity shell /.../ ....
Failed to find loop device:
could not attach image file to loop device:
no loop devices available
```
https://github.com/sylabs/singularity-cri/issues/373

This makes running .sif images locally difficult.
A workaround is to build development images with --sandbox flag,
to avoid using loopback devices. Production images can be built
without sandbox mode, to produce a .sif file.