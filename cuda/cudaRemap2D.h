/*
 * rovernet
 */

#ifndef __CUDA_REMAP_H
#define __CUDA_REMAP_H


#include "cudaUtility.h"


/**
 * For each pixel in the output image, the map contains the look-up coordinates into the input image.
 */
cudaError_t cudaRemap2D( float* input, short2* map, float* output, size_t width, size_t height );



#endif
