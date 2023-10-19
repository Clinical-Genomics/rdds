# Ubuntu 20.04/AMD64
FROM ubuntu@sha256:b795f8e0caaaacad9859a9a38fe1c78154f8301fdaf0872eaf1520d66d9c0b98 AS ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget

FROM gpubase as base

# Install OS deps
RUN apt-get install -y \
    xorg \
    xauth

# Install Conda
WORKDIR /tmp

ENV CONDAVER=py38_23.3.1-0-Linux-x86_64

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-$CONDAVER.sh && \
  sha256sum Miniconda3-$CONDAVER.sh | grep d1f3a4388c1a6fd065e32870f67abc39eb38f4edd36c4947ec7411e32311bd59 && \
  chmod +x Miniconda3-$CONDAVER.sh && \
  ./Miniconda3-$CONDAVER.sh -b -p /opt/conda && \
  rm Miniconda3-$CONDAVER.sh

# Install python deps
COPY src/requirements-conda.txt /tmp
COPY src/requirements-pip.txt /tmp
RUN . /opt/conda/bin/activate && \
  conda install -y --file /tmp/requirements-conda.txt && \
  pip3 install -r /tmp/requirements-pip.txt

# Setup prompt
RUN echo "export PS1=\"\[\033[1;31m\]\u@RD-DSDev:\w>\033[0m\]$\"" >> /root/.bashrc

# Setup aliases
RUN echo "alias ls=\"ls -lah --color=always\"" >> /root/.bashrc

FROM ubuntu AS motd
# Setup a login message via motd
RUN apt-get install -y figlet && \
  figlet -f slant RD-DSDev >> /usr/share/base-files/motd

FROM ubuntu AS dropbear
# Build dropbear SSH server to enable X11 forwarding
# Install source code and build tools
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y git build-essential
RUN mkdir -p /opt/dropbear
RUN git clone https://github.com/mkj/dropbear.git /opt/dropbear
WORKDIR /opt/dropbear
RUN git checkout DROPBEAR_2022.83

# Apply patches to disable root-privileged commands in container
COPY build/devenv/dropbear/disable-pty-chown.patch .
COPY build/devenv/dropbear/disable-setuid.patch .
RUN git apply disable-pty-chown.patch
RUN git apply disable-setuid.patch

# Enable X11 build flag, disable syslogging (to std out direct) and zlib
RUN echo "#define DROPBEAR_X11FWD 1" >> localoptions.h
RUN ./configure \
  --disable-syslog \
  --disable-zlib

# Build dropbear server and keygen tool
RUN make -j 4 PROGRAMS="dropbear dropbearkey"

FROM ubuntu AS pycharm
ENV PYCHARMVER=community-2023.1.2
RUN mkdir -p /opt
WORKDIR /opt
RUN wget https://download.jetbrains.com/python/pycharm-$PYCHARMVER.tar.gz && \
  sha256sum pycharm-$PYCHARMVER.tar.gz| grep 1445b48b091469176644cb85a0a6f953783920fb1ec9a53bcbdd932ad8c947b0
RUN tar -xf pycharm-$PYCHARMVER.tar.gz && rm pycharm-$PYCHARMVER.tar.gz
RUN ln -s pycharm-$PYCHARMVER pycharm

FROM base AS devenv
RUN apt-get update && \
 apt-get install -y --no-install-recommends \
 vim \
 git \
 bcftools \
 tabix

# Install dropbear SSH server and keygen tool
COPY --from=dropbear /opt/dropbear/dropbear /usr/bin
COPY --from=dropbear /opt/dropbear/dropbearkey /usr/bin

# Install login greeting message
COPY --from=motd /usr/share/base-files/motd /usr/share/base-files/motd
RUN echo "* Conda installation available at /opt/conda" >> /usr/share/base-files/motd

# Install pycharm
COPY --from=pycharm /opt/pycharm /opt/pycharm
RUN echo "* Pycharm installed into /opt/pycharm" >> /usr/share/base-files/motd

# Install client pubkey so that one can login via SSH
COPY build/devenv/devenv-docker.rsakey.pub /root/.ssh/authorized_keys
RUN chmod go-rw /root/.ssh/authorized_keys
RUN mkdir -p /etc/dropbear

# Create mountpoint directory for this repository
RUN mkdir -p /rdds
ENV PYTHONPATH=$PYTHONPATH:/rdds
RUN echo "* Repo mounted at /rdds" >> /usr/share/base-files/motd

# Install container test files
COPY build/devenv/guitest.py /

# Start dropbear SSHD on port 2150
RUN printf "#!/bin/bash\n\ndropbear -p 2150 -b /usr/share/base-files/motd -s -F -R" > /entrypoint.sh && \
  chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
