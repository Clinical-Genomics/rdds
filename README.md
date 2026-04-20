# RDDS: Data Science for Rare Disease Diagnostics

[![Tests](https://github.com/Clinical-Genomics/rdds/actions/workflows/test.yml/badge.svg)](https://github.com/Clinical-Genomics/rdds/actions/workflows/test.yml)
[![CI-CD](https://github.com/Clinical-Genomics/rdds/actions/workflows/build-test-push.yml/badge.svg)](https://github.com/Clinical-Genomics/rdds/actions/workflows/build-test-push.yml)

This repository contains tools for data exploration and
machine learning models targeting rare disease WGS analysis.

Provides a development environment that can be used both locally
as well as on SLURM cluster (using singularity, podman).

## Layout
```raw
.
├── build (repo infrastructure)
├── docs (documentation)
├── Makefile (main entrypoint for infrastructure, development work)
├── README.md (this file)
├── src (repo source code)
└── tmp (temporary output directory for modules)

./src
├── rdds
├── requirements-pip.txt
└── tests

./rdds
├── dataset_clinvar
├── dataset_giab
├── dataset_mutacc
├── gicam
├── lib
└── variant_rank_score (MIVMIR)
