#include <mpi.h>                  // MPI for distributed processing
#include <cuda_runtime.h>        // CUDA runtime for GPU operations
#include <iostream>              // Standard C++ I/O
#include "image_processing.cuh"  // Header for custom CUDA kernels (e.g., sobel_filter)
#include "clockcycle.h"          // Timing utility for POWER9 cycle counts

// Define constants for image dimensions and image size
#define IMAGE_SIZE 256*256            // Total pixels per image
#define WIDTH 256                     // Image width
#define HEIGHT 256                    // Image height
#define WEAK_SCALING_IMAGES_PER_RANK 8000  // In weak scaling, each rank processes this many images
#define STRONG_SCALING_IMAGES_PER_RANK 64000  // In strong scaling, this many images are processed

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);  // Initialize MPI environment
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Determine if running in weak or strong scaling mode
    bool weak_scaling = atoi(argv[1]);  // argv[1] should be 0 or 1

    // Determine how many images each rank will process
    // Weak scaling: fixed number per rank
    // Strong scaling: total images divided across ranks
    int images_per_rank = weak_scaling ? WEAK_SCALING_IMAGES_PER_RANK : STRONG_SCALING_IMAGES_PER_RANK / size;
    uint64_t io_start, io_end;
    io_start = clock_now();

    // Open input and output files in parallel using MPI I/O
    MPI_File input_file, output_file;
    MPI_File_open(MPI_COMM_WORLD, "input_data.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_open(MPI_COMM_WORLD, "output_data.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);

    // Allocate pinned host memory for input and output image buffers
    unsigned char* h_images = new unsigned char[images_per_rank * IMAGE_SIZE];
    unsigned char* h_results = new unsigned char[images_per_rank * IMAGE_SIZE];

    // Calculate file offset for this rank to read its portion
    MPI_Offset offset = rank * images_per_rank * IMAGE_SIZE;

    // Each rank reads its image chunk from the input file collectively
    MPI_File_read_at_all(input_file, offset, h_images, images_per_rank * IMAGE_SIZE, MPI_BYTE, MPI_STATUS_IGNORE);

    // Allocate memory on the GPU for input and output image buffers
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, images_per_rank * IMAGE_SIZE);
    cudaMalloc(&d_output, images_per_rank * IMAGE_SIZE);

    // Create a CUDA stream to allow asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing variables for I/O measurement
    double io_time = 0;

    // GPU processing loop for each image
    for (int i = 0; i < images_per_rank; ++i) {
        // Asynchronously copy one image from host to device
        cudaMemcpyAsync(&d_input[i * IMAGE_SIZE], &h_images[i * IMAGE_SIZE],
                        IMAGE_SIZE, cudaMemcpyHostToDevice, stream);

        // Launch the Sobel filter kernel on the image
        dim3 blocks(8, 8);       // 8x8 grid of blocks
        dim3 threads(32, 32);    // 32x32 threads per block (covers 256x256 image)
        sobel_filter<<<blocks, threads, 0, stream>>>(
            &d_input[i * IMAGE_SIZE], &d_output[i * IMAGE_SIZE], WIDTH, HEIGHT);

        // Asynchronously copy processed image back to host
        cudaMemcpyAsync(&h_results[i * IMAGE_SIZE], &d_output[i * IMAGE_SIZE],
                        IMAGE_SIZE, cudaMemcpyDeviceToHost, stream);
    }

    // Wait for all GPU operations to finish before moving on
    cudaStreamSynchronize(stream);

    // Start timing for I/O

    // Write all processed images to output file collectively
    MPI_File_write_at_all(output_file, offset, h_results,
                          images_per_rank * IMAGE_SIZE, MPI_BYTE, MPI_STATUS_IGNORE);

    io_end = clock_now();  // End timing

    // Free GPU and CPU resources
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_images;
    delete[] h_results;
    MPI_File_close(&input_file);
    MPI_File_close(&output_file);

    // Only rank 0 prints timing stats
    if (rank == 0) {
        io_time = (double)(io_end - io_start) / 512000000.0;  // Convert cycles to seconds
        std::cout << "Total Time: " << io_time << "s\n";
    }

    MPI_Finalize();  // Clean up MPI environment
    return 0;
}
