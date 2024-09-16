#!/bin/bash
# Start containerized development environment as a SLURM job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=15G
#SBATCH --time=08:00:00
#SBATCH --account=development
#SBATCH --job-name=RDDSdev
#SBATCH --qos=normal

make devenv-ubuntu_20_04-singularity-sshd
