/*
 * rovernet
 */

#include "cudaUtility.h"
#include "cudaMath.h"



// one thread is launched for each pixel in the output image
__global__ void gpuRemap2D( float* input, short2* map, float* output, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const int    index   = y * width + x;
	const short2 mapping = map[index];
	
	float value = 0.0;
	
	// only look-up the texture if the mapping coordinates are in-bounds 
	if( mapping.x >= 0 && mapping.x < width && mapping.y >= 0 && mapping.y < height )
		value = input[mapping.y * width + mapping.x];
	
	//printf("pixel %i, %i => %i, %i\n", x, y, (int)mapping.x, (int)mapping.y);		// you can printf from a CUDA kernel
	
	// the map goes from output pixels coordinates into input pixel coordinates
	output[index] = value;
}


cudaError_t cudaRemap2D( float* input, short2* map, float* output, size_t width, size_t height )
{
	if( !input || !output || map )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 )
		return cudaErrorInvalidValue;
	
	const dim3 blockDim(8, 8);		// the number of threads per block (64)
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuRemap2D<<<gridDim, blockDim>>>(input, map, output, width, height);
	
	return CUDA(cudaGetLastError());
}
