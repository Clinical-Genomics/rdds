#!/bin/bash

VERSION=25.04.3
wget https://github.com/nextflow-io/nextflow/releases/download/v25.04.3/nextflow-$VERSION-dist
sha256sum nextflow-$VERSION-dist | grep 53c232cdd8a9419d2c205dc7c6c4dd2646182c997300e6439a453099e28aa21a

