# Utilising NVIDIA GPUs in Containers

In general, the host kernel exposes the nvidia GPU to the container OS.
There's generally no need to install the GPU driver itself, as long as the host
is capable of allocating the GPU.

However, ecosystem libraries related to the driver must match the NVIDIA driver,
both in the host OS and container.

https://forums.developer.nvidia.com/t/cuda-12-1-tensorflow-on-linux-20-04-does-not-see-gpu-nvidia-4090-rtx/252514

https://discuss.tensorflow.org/t/gpu-with-cuda-11-8-not-detected-could-not-find-cuda-drivers/16085

## Ubuntu - NVIDIA GPU Drivers Repo
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
apt-get install -y linux-modules-nvidia-520-generic nvidia-driver-520
```
Should install `nvidia-driver-470, linux-modules-nvidia-470-5.15.0-86-generic`

### Configuring Ubuntu 20.04
```
ubuntu-drivers install nvidia:470
```

Be aware of `meta`-packages that install higher versions of nvidia drivers!
Make sure to read in on the package details; `apt-cache show nvidia-driver-470`.

## NVIDIA driver repository
https://www.nvidia.com/en-us/drivers/unix/
https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/

## Tensorflow CUDA - CUDNN tested versions
https://www.tensorflow.org/install/source#gpu

## NVIDIA GPU driver - CUDA lib compatibility
https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward

## cuDNN
https://developer.nvidia.com/cudnn
Easiest to install from tarball:
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar

Make sure you're logged in to NVIDIA cuDNN webpage; `https://developer.nvidia.com/rdp/cudnn-archive` (signup-walled)
```
$ wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
$ tar -xf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

There's a local deb installer too, but does not seem as easy to use:
https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.6.0.163_1.0-1_amd64.deb


## Tensorflow v2.12 GPU Requirements
```
NVIDIA® GPU drivers version 450.80.02 or higher.
CUDA® Toolkit 11.8.
cuDNN SDK 8.6.0.
```

## Making use of installed CUDA and cuDNN libraries
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## Testing
cat /proc/driver/nvidia/version
