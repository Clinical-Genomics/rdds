# Dockerfile to run Cosmograph network graph plots
ARG VERSION
FROM clinicalgenomics/rdds_nvidia:${VERSION}

# Install Python3.10 which is a dependency by Cosmograph
WORKDIR /opt/python3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz  && \
    gunzip Python-3.10.0.tgz && \
    tar -xf Python-3.10.0.tar
WORKDIR /opt/python3.10/Python-3.10.0

RUN apt-get update && \
    apt-get install -y \
    zlib1g-dev \
    libssl-dev \
    libsqlite3-dev \
    libbz2-dev \
    libncurses-dev \
    libjpeg-dev \
    libffi-dev \
    liblzma-dev

RUN ./configure && \
    make -j 2 && \
    make install

RUN python3.10 -m venv /opt/pyenv3.10 && \
    . /opt/pyenv3.10/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install cosmograph==0.0.38 \
    jupyterlab==4.3.4 \
    matplotlib==3.10.0 \
    tensorflow==2.12.0 \
    tensorflow_text==2.12.1 \
    seaborn==0.12.2

# TODO: entrypoint to start jupyter lab