#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1