/*
 * rovernet
 */

#include "cudaUtility.h"
#include "cudaMath.h"



// gpuResample
__global__ void gpuRangeMap( float* input, float* output, int oPitch, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const float angle = atan2f(y, x);

	printf("%i %i  %f\n", angle);
	//const float px = input[ dy * iPitch + dx ];

	//output[y*oPitch+x] = px;
}


// cudaResize
cudaError_t cudaRangeMap2D( float* input, float* output, size_t outputPitch, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( outputPitch == 0 || outputWidth == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth/blockDim.x, outputHeight/blockDim.y));

	gpuRangeMap<<<gridDim, blockDim>>>(input, output, outputPitch, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}