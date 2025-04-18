// image_processing.cuh
#ifndef IMAGE_PROCESSING_CUH
#define IMAGE_PROCESSING_CUH

__global__ void sobel_filter(unsigned char* input, 
                            unsigned char* output,
                            int width, int height);

#endif