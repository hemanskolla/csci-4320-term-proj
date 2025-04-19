# csci-4320-term-proj

# Workflow:

Compile with make clean && make

# Weak Scaling:

For each rank, run:

python input_data.py data/ input_data.dat --max-images (rank \* 10)

after moving the output_data.dat to local run:
python output_data.py output_data.dat (output folder name)

# 1 Rank strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 2 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 4 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 8 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 16 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 32 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 64 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=16
#SBATCH --ntasks=64
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# Strong Scaling

Before starting run:

python input_data.py data/ input_data.dat --max-images 640

after moving the output_data.dat to local run:
python output_data.py output_data.dat (output folder name)

# 1 Rank weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 2 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 4 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 8 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 16 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 32 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --partition=el8-rpi

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 64 Ranks weak.sh:

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

mpirun --bind-to core -np $SLURM_NTASKS ./main 0
