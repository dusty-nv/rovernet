/*
 * rovernet
 */

#include "cudaUtility.h"



// gpuResample
__global__ void gpuResize( float2 scale, float* input, int iPitch, float* output, int oPitch, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = (float)x * scale.x;
	const int dy = (float)y * scale.y;

	const float px = input[ dy * iPitch + dx ];

	output[y*oPitch+x] = px;
}


// cudaResize
cudaError_t cudaResize( float* input, size_t inputPitch, size_t inputWidth, size_t inputHeight,
				    float* output, size_t outputPitch, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputPitch == 0 || outputPitch == 0 || inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(outputWidth/blockDim.x, outputHeight/blockDim.y);

	gpuResize<<<gridDim, blockDim>>>(scale, input, inputPitch, output, outputPitch, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


