#!/bin/bash
# Start containerized development environment as a SLURM job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --account=development
#SBATCH --job-name=RDDSdev
#SBATCH --qos=normal

make devenv-singularity-sshd
