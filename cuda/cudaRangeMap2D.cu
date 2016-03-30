/*
 * rovernet
 */

#include "cudaUtility.h"
#include "cudaMath.h"



// gpuRangeMap
__global__ void gpuRangeMap( float* input, float* output, int oPitch, int oWidth, int oHeight, float range_scale )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const float angle    = atan2f(y, x);
	const float distance = sqrtf(x*x+y*y);
	
	printf("%f %f\n", angle, distance);
	
	const float range_sample = input[(int)angle] * range_scale;
	
	output[y*oPitch+x] = distance < range_sample ? 1.0f : 0.0f;
}


// cudaResize
cudaError_t cudaRangeMap2D( float* input, float* output, float maxRange, size_t outputPitch, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( outputPitch == 0 || outputWidth == 0 || outputHeight == 0 || outputWidth != outputHeight )
		return cudaErrorInvalidValue;

	const float range_scale = (float(outputWidth) * 0.5f) / maxRange;
	
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth/blockDim.x, outputHeight/blockDim.y));

	gpuRangeMap<<<gridDim, blockDim>>>(input, output, outputPitch, outputWidth, outputHeight, range_scale);

	return CUDA(cudaGetLastError());
}