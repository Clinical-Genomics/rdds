# Image containing cuDNN drivers not readily accessible over Internet due to Nvidia login-wall.
# In order to build this docker image, you must login to Nvidia homepage and
# download the tarballs. Then run the appropriate Makefile target.
# See: https://developer.nvidia.com/cudnn
# and https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar
FROM scratch

WORKDIR /opt/cudnn

# https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
COPY build/devenv/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz .
