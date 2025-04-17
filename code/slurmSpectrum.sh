#!/bin/bash
#SBATCH -N 1                  # Number of nodes
#SBATCH --partition=el8-rpi   # Required partition
#SBATCH --gres=gpu:4          # Request 4 GPUs per node
#SBATCH -t 30                 # Time limit (30 minutes max)
#SBATCH -J imgproc            # Job name
#SBATCH -o %x-%j.out          # Output file
#SBATCH -e %x-%j.err          # Error file

module load xl_r spectrum-mpi cuda/11.2

mpirun --bind-to core --report-bindings \
       -np $SLURM_NPROCS \
       ./your_executable input.raw output.raw