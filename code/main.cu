#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernels.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //input and output file names
    const char* input_filename = "data/input_image_data.bin";
    const char* output_filename = "data/output_image_data.bin";

    //number of bytes to read/write per process
    size_t local_data_size = 1024; 
    MPI_Offset offset = world_rank * local_data_size;

    //allocate host memory
    unsigned char *h_data = (unsigned char*)malloc(local_data_size);
    if (!h_data) {
        fprintf(stderr, "Rank %d: Error allocating host memory\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //mpi i/o: read input data
    MPI_File mpi_in;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &mpi_in);
    MPI_File_read_at(mpi_in, offset, h_data, local_data_size, MPI_BYTE, &status);
    MPI_File_close(&mpi_in);

    //run CUDA kernel for image processing
    unsigned char *d_data;
    cudaMalloc((void**)&d_data, local_data_size);
    cudaMemcpy(d_data, h_data, local_data_size, cudaMemcpyHostToDevice);

    //launch the kernel
    launchImageProcessingKernel(d_data, local_data_size);

    //wait for kernel to finish and copy the results back to host
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, local_data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    //mpi i/o: write output data
    MPI_File mpi_out;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpi_out);
    MPI_File_write_at(mpi_out, offset, h_data, local_data_size, MPI_BYTE, &status);
    MPI_File_close(&mpi_out);

    //free
    free(h_data);
    MPI_Finalize();
    return 0;
}
