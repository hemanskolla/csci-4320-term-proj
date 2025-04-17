#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <cuda_runtime.h>

void launchGaussianBlur(unsigned char* d_in,
                        unsigned char* d_out,
                        int width, int height,
                        cudaStream_t stream);

void launchSobelEdge(unsigned char* d_in,
                     unsigned char* d_out,
                     int width, int height,
                     cudaStream_t stream);

#endif