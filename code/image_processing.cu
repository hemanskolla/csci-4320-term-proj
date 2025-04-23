#include "image_processing.cuh"  // Declaration for sobel_filter
#include <cuda_runtime.h>        // CUDA runtime API
#include <math.h>                // For sqrtf()

/**
 * sobel_filter kernel
 *
 * Applies a 3×3 Sobel edge‐detection filter to a grayscale image.
 * Each CUDA thread computes one output pixel.
 *
 * Parameters:
 *   input  – pointer to input image data (1 byte per pixel)
 *   output – pointer to output image data (1 byte per pixel)
 *   width  – image width in pixels
 *   height – image height in pixels
 */
__global__ void sobel_filter(unsigned char* input,
                             unsigned char* output,
                             int width,
                             int height)
{
    // Compute the 2D pixel coordinates (x,y) for *this* thread
    // blockIdx   = which block in the grid
    // blockDim   = how many threads per block (x and y)
    // threadIdx  = which thread in the block
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only process *interior* pixels (skip 1‐pixel border),
    // since Sobel uses neighbors at x±1, y±1
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Compute horizontal gradient Gx using Sobel kernel:
        //
        //    [ -1   0   +1 ]
        //    [ -2   0   +2 ]
        //    [ -1   0   +1 ]
        //
        int gx = 
            - input[(y - 1) * width + (x - 1)]   // top‐left
            - 2 * input[y * width + (x - 1)]     // middle‐left
            -   input[(y + 1) * width + (x - 1)] // bottom‐left
            +   input[(y - 1) * width + (x + 1)] // top‐right
            + 2 * input[y * width + (x + 1)]     // middle‐right
            +   input[(y + 1) * width + (x + 1)]; // bottom‐right
        
        // Compute vertical gradient Gy using Sobel kernel:
        //
        //    [ -1  -2  -1 ]
        //    [  0   0   0 ]
        //    [ +1  +2  +1 ]
        //
        int gy = 
            - input[(y - 1) * width + (x - 1)]   // top‐left
            - 2 * input[(y - 1) * width + x]     // top‐center
            -   input[(y - 1) * width + (x + 1)] // top‐right
            +   input[(y + 1) * width + (x - 1)] // bottom‐left
            + 2 * input[(y + 1) * width + x]     // bottom‐center
            +   input[(y + 1) * width + (x + 1)]; // bottom‐right
        
        // Compute gradient magnitude: sqrt(gx^2 + gy^2)
        // Clamp to 255 to fit in a byte
        float mag = sqrtf((float)(gx * gx + gy * gy));
        if (mag > 255.0f) mag = 255.0f;
        
        // Write result back to output image
        output[y * width + x] = (unsigned char)mag;
    }
    // Threads whose (x,y) are on the border do nothing (leaving output undefined there).
}
