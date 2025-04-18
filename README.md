# csci-4320-term-proj

# Workflow:

Compile with make clean && make

# Weak Scaling:

# 1 Rank:

python input_data.py data/ weak_10.dat --max-images 10

salloc -N 1 --partition=el8-rpi --gres=gpu:1 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=1 --ntasks-per-node=1 ./main 1

# 2 Ranks:

python input_data.py data/ weak_20.dat --max-images 20

salloc -N 1 --partition=el8-rpi --gres=gpu:2 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=2 --ntasks-per-node=2 ./main 1

# 4 Ranks:

python input_data.py data/ weak_40.dat --max-images 40

salloc -N 1 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=4 --ntasks-per-node=4 ./main 1

# 8 Ranks:

python input_data.py data/ weak_80.dat --max-images 80

salloc -N 2 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 2 --ntasks=8 --ntasks-per-node=4 ./main 1

output_data.dat should have been created

# 16 Ranks:

python input_data.py data/ weak_10.dat --max-images 10

salloc -N 1 --partition=el8-rpi --gres=gpu:1 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=1 --ntasks-per-node=1 ./main 1

# 32 Ranks:

python input_data.py data/ weak_320.dat --max-images 320

salloc -N 8 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 8 --ntasks=32 --ntasks-per-node=4 ./main 1

# 64 Ranks:

python input_data.py data/ weak_640.dat --max-images 640

salloc -N 16 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 16 --ntasks=64 --ntasks-per-node=4 ./main 1

# Strong Scaling

# 1 Rank:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 1 --partition=el8-rpi --gres=gpu:1 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=1 --ntasks-per-node=1 ./main 0

# 2 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 1 --partition=el8-rpi --gres=gpu:2 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=2 --ntasks-per-node=2 ./main 0

# 4 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 1 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 1 --ntasks=4 --ntasks-per-node=4 ./main 0

# 8 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 2 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 2 --ntasks=8 --ntasks-per-node=4 ./main 0

# 16 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 4 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 4 --ntasks=16 --ntasks-per-node=4 ./main 0

# 32 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 8 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 8 --ntasks=32 --ntasks-per-node=4 ./main 0

# 64 Ranks:

python input_data.py data/ strong_640.dat --max-images 640

salloc -N 16 --partition=el8-rpi --gres=gpu:4 -t 10

module load xl_r spectrum-mpi cuda/11.2
srun -N 16 --ntasks=64 --ntasks-per-node=4 ./main 0
