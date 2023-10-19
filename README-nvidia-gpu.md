# Utilising NVIDIA GPUs in Containers

This readme is about utilising GPU from inside of Docker and Singularity
containers.

It's targeting Tensorflow dependencies (CUDA, cuDNN libraries).
In general, Tensorflow is only tested, compatible with a subset of Nvidia GPU
drivers and related libraries. The Tensorflow GPU functionality
is also coupled to these libraries at build time. Every Tensorflow version
is only compatible with a specific Nvidia GPU Driver and accompanying libraries.

Tensorflow `v2.12` has the following requirements:
```
NVIDIA® GPU drivers version 450.80.02 or higher.
CUDA® Toolkit 11.8.
cuDNN SDK 8.6.0.
```

See [tested GPU versions](https://www.tensorflow.org/install/source#gpu) for more information.

## Host - Container Integration
In general, the host kernel exposes the Nvidia GPU to the container OS.
It's the host (Nvidia) kernel (driver) that determines what driver version
to use.

From inside the container, there's strictly no need to install the GPU driver
itself, but the driver package often comes with an ecosystem that's required
for other applications. Therefore, it's often less problematic to mirror
the host Nvidia driver version into the container.

However, ecosystem libraries related to the driver must match the Nvidia driver,
both in the host OS and container.

### A Note On Docker GPU Ecosystem
Since we're targeting a SLURM environment, we cannot depend on
Nvidia's GPU integration suite, `Nvidia Container Toolkit` and
`Nvidia Container Runtime` which is a Docker-centered framework that solves GPU integration hassle.

There are also Nvidia built GPU enabled docker images available upstream.
However, basing our images on these Nvidia provided images would make it
difficult to truly custom-build our ecosystem.

### Downstream Libraries Depends On Nvidia GPU Driver
Libraries such as Nvidia CUDA and Nvidia cuDNN are dependent on the Nvidia GPU
Driver version.

See [this compatibility matrix](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward)
for more information.

## Host Setup
### Configuring Ubuntu 20.04
#### ubuntu-drivers
https://ubuntu.com/server/docs/nvidia-drivers-installation

The preferred way is to install Nvidia GPU driver using the `ubuntu-drivers` tool.
```
ubuntu-drivers install nvidia:470
```

#### Ubuntu NVIDIA GPU Drivers Dist
Installing (old) NVIDIA driver not available via the `ubuntu-drivers` utility,
via APT utility.

```
Installing the NVIDIA driver manually means installing the
correct kernel modules first, then installing the metapackage for the driver series.
```

Useful for customizing the install w.r.t Nvidia Driver - CUDA versions.
https://ubuntu.com/server/docs/nvidia-drivers-installation

Example
```
apt-get install -y linux-modules-nvidia-470-generic nvidia-driver-470
```
Should install `nvidia-driver-470, linux-modules-nvidia-470-5.15.0-86-generic`

Be aware of `meta`-packages that install higher versions of nvidia drivers!
Make sure to read in on the package details; `apt-cache show nvidia-driver-470`.

#### NVIDIA Driver Repository
Installing from .run scripts from Nvidia is also an alternative,
but it's very cumbersome and prone to configuration errors.

https://www.nvidia.com/en-us/drivers/unix/
https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/

## Nvidia CUDA
CUDA library is a development library to allow general purpose computing
on Nvidia's GPUs.

Nvidia provides a remote .deb Ubuntu installer.

See [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
for more information.

## Nvidia cuDNN
Nvidia CUDA Deep Neural Network (cuDNN) library is an optimized library
for DNN applications on Nvidia GPUs.

See [cuDNN Introduction](https://developer.nvidia.com/cudnn) and
[cuDNN Linux Install Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar)
for more information.

Access to the library is "account-walled" by Nvidia.

Since these tarballs are not ready accessible over HTTPS, there's a
`clinicalgenomics` image built for CI/CD purposes, see
[nvidia-cudnn.Dockerfile](build/devenv/nvidia-cudnn.Dockerfile).

Downloading of the library can be done by accessing the
[cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive), a full URL example:
`https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz`.

There's a local deb installer too, but there it's not as easy to select install flavor
as in the tarball case.

## Installation Process
0. Install Nvidia Driver to Host machine

Install CUDA and cuDNN libraries in container, do;
1. Install Nvidia GPU Driver
2. Install CUDA
3. Install cuDNN into CUDA library directory
4. Setup `PATH`, `LD_LIBRARY_PATH` to allow access to tooling and dynamic linking:
```
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Testing
```
cat /proc/driver/nvidia/version
find / -name cuda* 2>/dev/null
```
and additionally run the [python GPU test script](build/devenv/gputest.py).
