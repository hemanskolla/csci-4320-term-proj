CC = mpicxx
NVCC = nvcc
CFLAGS = -O3 -std=c++17
CUDAFLAGS = -arch=sm_70 -Xcompiler="-O3" -I$(MPI_ROOT)/include
LDFLAGS = -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_ROOT)/lib -lmpi_ibm

all: main

main: main.o image_processing.o
	$(CC) $^ -o $@ $(LDFLAGS)

main.o: main.cu image_processing.cuh
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

image_processing.o: image_processing.cu image_processing.cuh
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

clean:
	rm -f *.o main