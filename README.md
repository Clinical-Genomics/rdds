# RDDS: Data Science for Rare Disease Diagnostics

This repository contains tools for data exploration and
machine learning development targeting rare disease analysis.

This repo provides a development environment that can be used both locally
as well as on SLURM.

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
