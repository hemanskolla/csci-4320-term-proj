#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"

// Timing function (now used)
static __inline__ unsigned long long getticks(void) {
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__("mftb %0" : "=r"(tbl));
        __asm__ __volatile__("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((unsigned long long)tbu0) << 32) | tbl;
}

// Updated halo exchange declaration
void exchange_halos(unsigned char* local_data, int width, int height,
                    int rank, int world_size);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (rank == 0) fprintf(stderr, "Usage: %s <input.raw> <output.raw>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Timing and dimensions setup (unchanged)
    unsigned long long start_time, end_time;
    double total_time = 0.0;
    int global_width, global_height;
    if (rank == 0) {
        char header_file[256];
        snprintf(header_file, sizeof(header_file), "%s.header", argv[1]);
        FILE *header = fopen(header_file, "r");
        fscanf(header, "%d %d", &global_width, &global_height);
        fclose(header);
    }
    MPI_Bcast(&global_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate local chunk dimensions (unchanged)
    int local_rows = global_height / world_size;
    int remainder = global_height % world_size;
    int local_height = local_rows + (rank < remainder ? 1 : 0) + 2;
    unsigned char* h_data = (unsigned char*)malloc(global_width * local_height * sizeof(unsigned char));
    memset(h_data, 0, global_width * local_height);

    // Calculate MPI-IO offsets using prefix sum
    int local_data_size = (local_height - 2) * global_width;
    int offset = 0;
    int prefix_sum = 0;
    MPI_Exscan(&local_data_size, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank != 0) offset = prefix_sum;

    // MPI-IO Read
    start_time = getticks();
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at_all(fh,
                        offset,
                        h_data + global_width,  // Skip top halo
                        local_data_size,
                        MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    // Halo exchange (unchanged)
    exchange_halos(h_data, global_width, local_height-2, rank, world_size);

    // CUDA processing with separate buffers
    unsigned char *d_in, *d_out;
    size_t buffer_size = global_width * local_height * sizeof(unsigned char);
    cudaMalloc((void**)&d_in, buffer_size);
    cudaMalloc((void**)&d_out, buffer_size);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    start_time = getticks();
    cudaMemcpyAsync(d_in, h_data, buffer_size, cudaMemcpyHostToDevice, stream);

    // Process with separate buffers
    launchGaussianBlur(d_in + global_width, 
                      d_out + global_width,
                      global_width, local_height-2, stream);

    launchSobelEdge(d_out + global_width,
                   d_in + global_width,
                   global_width, local_height-2, stream);

    cudaMemcpyAsync(h_data, d_in, buffer_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    end_time = getticks();
    total_time = end_time - start_time;

    // MPI-IO Write with corrected offset
    MPI_File_open(MPI_COMM_WORLD, argv[2],
                 MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh,
                         offset,
                         h_data + global_width,
                         local_data_size,
                         MPI_BYTE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    // Cleanup
    free(h_data);
    cudaFree(d_in);
    cudaFree(d_out);
    
    if (rank == 0) {
        printf("Processing time: %llu ticks\n", total_time);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

// Updated halo exchange implementation
void exchange_halos(unsigned char* local_data, int width, int height,
                    int rank, int world_size) {
    MPI_Request reqs[4];
    int active = 0;

    // Top neighbor
    if (rank > 0) {
        MPI_Isend(local_data + width, width, MPI_BYTE, rank-1, 0, MPI_COMM_WORLD, &reqs[active++]);
        MPI_Irecv(local_data, width, MPI_BYTE, rank-1, 0, MPI_COMM_WORLD, &reqs[active++]);
    }

    // Bottom neighbor
    if (rank < world_size-1) {
        MPI_Isend(local_data + (height-2)*width, width, MPI_BYTE, rank+1, 0, MPI_COMM_WORLD, &reqs[active++]);
        MPI_Irecv(local_data + (height-1)*width, width, MPI_BYTE, rank+1, 0, MPI_COMM_WORLD, &reqs[active++]);
    }

    MPI_Waitall(active, reqs, MPI_STATUS_IGNORE);
}