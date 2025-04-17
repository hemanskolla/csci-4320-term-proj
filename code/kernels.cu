#include "kernels.h"
#include <cuda_runtime.h>

// 3×3 Gaussian kernel in constant memory
__constant__ float G[3][3] = {
    {1/16.f, 2/16.f, 1/16.f},
    {2/16.f, 4/16.f, 2/16.f},
    {1/16.f, 2/16.f, 1/16.f}
};

// Simple 2D block + shared‐memory tile for a 3×3 blur
__global__ void gaussianBlurKernel(
    unsigned char* d_in,
    unsigned char* d_out,
    int width, int height)
{
    extern __shared__ unsigned char tile[]; 
    int tx = threadIdx.x, ty = threadIdx.y;
    int x  = blockIdx.x * blockDim.x + tx;
    int y  = blockIdx.y * blockDim.y + ty;

    // load into tile with 1‐pixel border
    int tileW = blockDim.x + 2, tileH = blockDim.y + 2;
    int lx = tx + 1, ly = ty + 1;
    if (x < width && y < height) {
        tile[ly*tileW + lx] = d_in[y*width + x];
        // halo
        if      (tx == 0      && x > 0       ) tile[ly*tileW + 0] = d_in[y*width + (x-1)];
        else if (tx==blockDim.x-1 && x<width-1) tile[ly*tileW + lx+1] = d_in[y*width + (x+1)];
        if      (ty == 0      && y > 0       ) tile[0*tileW + lx] = d_in[(y-1)*width + x];
        else if (ty==blockDim.y-1 && y<height-1) tile[(ly+1)*tileW + lx] = d_in[(y+1)*width + x];
        // corners
        if (tx==0 && ty==0 && x>0 && y>0)
            tile[0*tileW + 0] = d_in[(y-1)*width + (x-1)];
        if (tx==0 && ty==blockDim.y-1 && x>0 && y<height-1)
            tile[(ly+1)*tileW + 0] = d_in[(y+1)*width + (x-1)];
        if (tx==blockDim.x-1 && ty==0 && x<width-1 && y>0)
            tile[0*tileW + lx+1] = d_in[(y-1)*width + (x+1)];
        if (tx==blockDim.x-1 && ty==blockDim.y-1 && x<width-1 && y<height-1)
            tile[(ly+1)*tileW + lx+1] = d_in[(y+1)*width + (x+1)];
    }
    __syncthreads();

    // apply 3×3 convolution
    if (x < width && y < height) {
        float sum = 0;
        #pragma unroll
        for (int ky=0; ky<3; ++ky)
            #pragma unroll
            for (int kx=0; kx<3; ++kx)
                sum += G[ky][kx] * tile[(ly + ky - 1)*tileW + (lx + kx - 1)];
        d_out[y*width + x] = (unsigned char)sum;
    }
}

// Sobel kernels in constant memory
__constant__ int Sx[3][3] = {
    {+1,  0, -1},
    {+2,  0, -2},
    {+1,  0, -1}
};
__constant__ int Sy[3][3] = {
    {+1, +2, +1},
    { 0,  0,  0},
    {-1, -2, -1}
};

__global__ void sobelKernel(
    unsigned char* d_in,
    unsigned char* d_out,
    int width, int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x<1 || y<1 || x>=width-1 || y>=height-1) return;

    float gx=0, gy=0;
    #pragma unroll
    for (int ky=0; ky<3; ++ky)
        #pragma unroll
        for (int kx=0; kx<3; ++kx) {
            float val = d_in[(y + ky -1)*width + (x + kx -1)];
            gx += Sx[ky][kx]*val;
            gy += Sy[ky][kx]*val;
        }
    float mag = sqrtf(gx*gx + gy*gy);
    d_out[y*width + x] = (unsigned char)(mag>255 ? 255 : mag);
}

void launchGaussianBlur(unsigned char* d_in,
                        unsigned char* d_out,
                        int width, int height,
                        cudaStream_t stream)
{
    dim3 block(16,16), grid(
        (width + block.x-1)/block.x,
        (height+ block.y-1)/block.y
    );
    size_t shmem = (block.x+2)*(block.y+2)*sizeof(unsigned char);
    gaussianBlurKernel<<<grid,block,shmem,stream>>>(d_in,d_out,width,height);
}

void launchSobelEdge(unsigned char* d_in,
                     unsigned char* d_out,
                     int width, int height,
                     cudaStream_t stream)
{
    dim3 block(16,16), grid(
        (width + block.x-1)/block.x,
        (height+ block.y-1)/block.y
    );
    sobelKernel<<<grid,block,0,stream>>>(d_in,d_out,width,height);
}
