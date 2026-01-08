ARG OS_FLAVOUR
ARG VERSION

FROM clinicalgenomics/rdds-ubuntu-20.04-nvidia:${VERSION} AS ubuntu_20_04_nvidia
# Placeholder to set stage label name

# Ubuntu 20.04/AMD64
FROM ubuntu@sha256:b795f8e0caaaacad9859a9a38fe1c78154f8301fdaf0872eaf1520d66d9c0b98 AS ubuntu_20_04

# Ubuntu 22.04/AMD64
FROM ubuntu@sha256:1c4cc37c10c4678fd5369d172a4e079af8a28a6e6f724647ccaa311b4801c3c9 AS ubuntu_22_04

FROM ${OS_FLAVOUR} as base
ENV DEBIAN_FRONTEND=noninteractive
# Install OS deps
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    wget \
    xorg \
    xauth \
    python3-venv \
    python3-tk \
    bcftools \
    tabix

# Install python deps
COPY src/requirements-pip.txt /tmp
RUN python3 -m venv /opt/pyenv
RUN . /opt/pyenv/bin/activate && \
  pip3 install --upgrade pip && \
  pip3 install -r /tmp/requirements-pip.txt

# Setup prompt
RUN echo "export PS1=\"\[\033[1;31m\]\u@RD-DSDev:\w>\033[0m\]$\"" >> /root/.bashrc

# Setup aliases
RUN echo "alias ls=\"ls -lah --color=always\"" >> /root/.bashrc

FROM base AS mivmir-production
RUN mkdir -p /rdds/src
COPY src /rdds/src
COPY <<EOF /entrypoint-mivmir.bash
#!/bin/bash
set -x
set -e
. /opt/pyenv/bin/activate
export PYTHONPATH=/rdds/src
python3 -m pytest /rdds/src/tests/variant_rank_score -k test_inference
python3 -m rdds.variant_rank_score predict-on-vcf \$@
EOF
RUN chmod +x /entrypoint-mivmir.bash

FROM base AS motd
# Setup a login message via motd
RUN apt-get install -y figlet && \
  figlet -f slant RD-DSDev >> /usr/share/base-files/motd

FROM base AS dropbear
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

FROM base AS pycharm
ENV PYCHARMVER=community-2025.2.1
RUN mkdir -p /opt
WORKDIR /opt
RUN wget https://download.jetbrains.com/python/pycharm-$PYCHARMVER.tar.gz && \
  sha256sum pycharm-$PYCHARMVER.tar.gz| grep fda3fef97cbc6591cee64fdc7e48bb7a5634be63e527293cd5146537dc562493
RUN tar -xf pycharm-$PYCHARMVER.tar.gz && rm pycharm-$PYCHARMVER.tar.gz
RUN ln -s pycharm-$PYCHARMVER pycharm

FROM base AS devenv
RUN apt-get update && \
 apt-get install -y --no-install-recommends \
 vim \
 git \
 htop

# Install dropbear SSH server and keygen tool
COPY --from=dropbear /opt/dropbear/dropbear /usr/bin
COPY --from=dropbear /opt/dropbear/dropbearkey /usr/bin

# Install login greeting message
COPY --from=motd /usr/share/base-files/motd /usr/share/base-files/motd
RUN echo "* Python venv installation available at /opt/pyenv" >> /usr/share/base-files/motd

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

FROM devenv AS devenv_pcaet
# https://docs.ollama.com/linux#installing-specific-versions
ENV OLLAMA_VERSION=0.13.4
WORKDIR /opt/ollama
RUN apt-get install -y curl && \
    curl -LO https://ollama.com/download/ollama-linux-amd64.tgz && \
    tar -xzf ollama-linux-amd64.tgz && \
    rm ollama-linux-amd64.tgz
COPY build/ollama/bootstrap_models.py .
RUN . /opt/pyenv/bin/activate && \
    python3 bootstrap_models.py