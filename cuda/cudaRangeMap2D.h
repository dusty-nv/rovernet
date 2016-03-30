/*
 * rovernet
 */

#ifndef __CUDA_RANGE_MAP_2D_H
#define __CUDA_RANGE_MAP_2D_H


#include "cudaUtility.h"


/**
 * Function that takes in LIDAR scan and produces a radial 2D map of the range data.
 * Pixels that are 0.0 correspond to areas that are within the reported range for that angle.
 * Pixels that are 1.0 correspond to areas that are outside the reported range for that angle.
 * @param input array of 360 range samples
 */
cudaError_t cudaRangeMap2D( float* input, float* output, float maxRange,
				            size_t outputPitch, size_t outputWidth, size_t outputHeight );



#endif