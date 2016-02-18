/*
 * rovernet
 */

#ifndef __ROVERNET_H_
#define __ROVERNET_H_


#include <stdio.h>
#include <stdint.h>


struct lua_State;
struct THCState;
struct THFloatTensor;
struct THCudaTensor;


/*
 * roverNet
 */
class roverNet
{
public:
	/**
	 * Create a new instance of roverNet.
	 */
	static roverNet* Create();
	
	/**
	 * Destructor
	 */
	~roverNet();

	/**
	 * Tensor wrapper for working with Torch/cuTorch.
	 */
	struct Tensor
	{
		THFloatTensor* cpuTensor;
		THCudaTensor*  gpuTensor;	// (THCudaTensor defined as THCudaFloatTensor in THCGenerateAllTypes)

		float* cpuPtr;
		float* gpuPtr;

		uint32_t width;
		uint32_t height;
		uint32_t depth;

		size_t elements;
		size_t size;
	};

	/**
	 * Allocate a Torch float tensor mapped to CPU/GPU.
	 */
	Tensor* AllocTensor( uint32_t width, uint32_t height=1, uint32_t depth=1 );

	/**
	 * Run the next iteration of the network.
	 */
	bool updateNetwork( Tensor* input, Tensor* goal, Tensor* output );

	/**
	 * Run the next iteration of the network.
	 */
	//bool updateNetwork( float* image, float* reward, float* output );	// eventually there will be API
															// that handles Tensors internally

private:
	roverNet();
	bool init();

	lua_State* L;		/**< Lua/Torch7 operating environment */
	THCState*  THC;	/**< cutorch state */
};


#endif
