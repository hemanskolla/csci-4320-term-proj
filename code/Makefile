CC = mpicxx
CFLAGS = -O3 -std=c++17
NVCC = nvcc
NVFLAGS = -arch=sm_70 -Xcompiler -O3 -I${MPI_ROOT}/include

all: imgproc

imgproc: main.o kernels.o
	$(CC) $(CFLAGS) $^ -o $@ -L${CUDA_HOME}/lib64 -lcudart

main.o: main.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

kernels.o: kernels.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f *.o imgproc