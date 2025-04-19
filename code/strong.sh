#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

# Add IBM MPI license agreement acceptance
export IBM_MPI_LICENSE_ACCEPT=yes

# Use Spectrum MPI's launcher wrapper
mpirun --bind-to core -np $SLURM_NTASKS ./main 0