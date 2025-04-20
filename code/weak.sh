#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1