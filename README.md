# csci-4320-term-proj

Depending on dataset change:

- #define IMAGE_SIZE 256*256 (img len * img width)
- #define WIDTH 256 (img width)
- #define HEIGHT 256 (img height)
- #define WEAK_SCALING_IMAGES_PER_RANK 8000 (how many images you want to process per rank)
- #define STRONG_SCALING_IMAGES_PER_RANK 64000 (the total number of images to be processed)

# Workflow:

Compile with make clean && make

# Weak Scaling:

For each rank, run:

python input_data.py data/ input_data.dat --max-images (rank \* 10)

after moving the output_data.dat to local run:
python output_data.py output_data.dat (output folder name)

# 1 Rank weak.sh:

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

# 2 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 4 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# 8 Ranks weak.sh:

#!/bin/bash
#SBATCH --job-name=img-weak
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 1

# Strong Scaling

Before starting run:

python input_data.py data/ input_data.dat --max-images 640

after moving the output_data.dat to local run:
python output_data.py output_data.dat (output folder name)

# 1 Rank strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 2 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 4 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0

# 8 Ranks strong.sh:

#!/bin/bash
#SBATCH --job-name=img-strong
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:01:00
#SBATCH --partition=el8

module purge
module load xl_r spectrum-mpi cuda/11.2

export IBM_MPI_LICENSE_ACCEPT=yes

mpirun --bind-to core -np $SLURM_NTASKS ./main 0
