/*
 * rovernet
 */

#ifndef __CUDA_RESIZE_H
#define __CUDA_RESIZE_H


#include "cudaUtility.h"


/**
 * Function for increasing or decreasing the size of an image on the GPU.
 */
cudaError_t cudaResize( float* input,  size_t inputPitch,  size_t inputWidth, size_t inputHeight,
				    float* output, size_t outputPitch, size_t outputWidth, size_t outputHeight );



#endif

