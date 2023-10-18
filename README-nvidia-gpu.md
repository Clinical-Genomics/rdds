# Utilising NVIDIA GPUs in Containers

In general, the host kernel exposes the nvidia GPU to the container OS.
There's generally no need to install the GPU driver itself, as long as the host
is capable of allocating the GPU.

However, ecosystem libraries related to the driver must match the NVIDIA driver,
both in the host OS and container.

https://forums.developer.nvidia.com/t/cuda-12-1-tensorflow-on-linux-20-04-does-not-see-gpu-nvidia-4090-rtx/252514

https://discuss.tensorflow.org/t/gpu-with-cuda-11-8-not-detected-could-not-find-cuda-drivers/16085

## NVIDIA driver repository
https://www.nvidia.com/en-us/drivers/unix/
https://www.nvidia.com/en-us/drivers/unix/linux-amd64-display-archive/

## Tensorflow CUDA - CUDNN tested versions
https://www.tensorflow.org/install/source#gpu

## NVIDIA GPU driver - CUDA lib compatibility
https://docs.nvidia.com/deploy/cuda-compatibility/index.html#deployment-consideration-forward

##
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

## Testing
cat /proc/driver/nvidia/version
