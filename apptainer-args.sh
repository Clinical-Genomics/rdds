#!/bin/bash
# Helper script to adjust singularity/apptainer cmdline for sandbox R/W access

set -e

VERSION=`singularity --version`


if echo $VERSION | grep apptainer >/dev/null; then
  echo "--unsquash"
elif echo $VERSION | grep singularity >/dev/null; then
  true
else
  echo "Unknown singularity version"
  exit 1
fi