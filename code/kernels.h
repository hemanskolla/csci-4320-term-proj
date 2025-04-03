#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

void launchImageProcessingKernel(unsigned char *d_data, size_t data_size);

#endif
