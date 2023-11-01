# Ubuntu 20.04/AMD64
FROM ubuntu@sha256:b795f8e0caaaacad9859a9a38fe1c78154f8301fdaf0872eaf1520d66d9c0b98 AS ubuntu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget \
    kmod

WORKDIR /tmp

# Install NVIDIA GPU driver
# Additionally install meta package linux-modules-nvidia-520-generic as well?
RUN apt-get install --no-install-recommends -y nvidia-driver-470=470.199.02-0ubuntu0.20.04.1

# Install NVIDIA CUDA library
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
  dpkg -i cuda-keyring_1.1-1_all.deb && \
  apt-get update && \
  apt-get install -y cuda-libraries-11-8 \
  cuda-compiler-11-8 && \
  rm *.deb

# Install NVIDIA cuDNN libraries into CUDA dir (must be compatible with Nvidia Driver, CUDA versions)
COPY --from=clinicalgenomics/rdds-nvidia-cudnn:draft /opt/cudnn/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz .
RUN tar -xf cudnn-*.tar.xz && \
  mkdir -p /usr/local/cuda/include && \
  cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include  && \
  cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64  && \
  chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*  && \
  rm *.tar.xz

RUN \
  echo "# Source this prior to initialising python environment" > /opt/init-cuda-cudnn && \
  echo "export PATH=/usr/local/cuda-11.8/bin:\$PATH" >> /opt/init-cuda-cudnn && \
  echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH" >> /opt/init-cuda-cudnn
