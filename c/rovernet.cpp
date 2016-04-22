/*
 * rovernet
 */

#include "rovernet.h"
#include <string.h>

extern "C" 
{ 
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
#include <luaT.h>
}

#include <THC/THC.h>
#include "../cuda/cudaMappedMemory.h"

#define SCRIPT_FILENAME "bearing2.lua"
//#define SCRIPT_FILENAME  "rovernet.lua"
#define SCRIPT_FUNC_NAME "update_network"



// constructor
roverNet::roverNet()
{
	L   = NULL;
	THC = NULL;
}


// destructor
roverNet::~roverNet()
{
	if( L != NULL )
	{
		lua_close(L);
		L = NULL;
	}
}


// Create
roverNet* roverNet::Create()
{
	roverNet* r = new roverNet();

	if( !r )
		return NULL;

	if( !r->init() )
	{
		printf("[rovernet]  failed to initialize roverNet\n");
		delete r;
		return NULL;
	}

	return r;
}


static roverNet::Tensor* new_Tensor()
{
	roverNet::Tensor* t = new roverNet::Tensor();

	t->cpuTensor = NULL;
	t->gpuTensor = NULL;
	t->cpuPtr    = NULL;
	t->gpuPtr    = NULL;
	t->size      = 0;

	return t;
}


// AllocTensor
roverNet::Tensor* roverNet::AllocTensor( uint32_t width, uint32_t height, uint32_t depth )
{
	const size_t elem = width * height * depth;
	const size_t size = elem * sizeof(float);

	if( size == 0 )
		return NULL;

	// create Tensor wrapper object
	Tensor* t = new_Tensor();
	   

	// alloc CUDA mapped memory
	if( !cudaAllocMapped((void**)&t->cpuPtr, (void**)&t->gpuPtr, size) )
	{
		printf("[rovernet]  failed to alloc CUDA buffers for tensor size %zu bytes\n", size);
		return NULL;
	}


#if 0
	// set memory to default sequential pattern for debugging
	for( size_t n=0; n < elem; n++ )
		t->cpuPtr[n] = float(n);
#endif


	// alloc CPU tensor
	THFloatStorage* cpuStorage = THFloatStorage_newWithData(t->cpuPtr, elem);

	if( !cpuStorage )
	{
		printf("[rovernet]  failed to alloc CPU THFloatStorage\n");
		return NULL;
	}

	long sizedata[2]   = { height, width };		// BUG:  should be reversed?
	long stridedata[2] = { width, 1 };	// with YUV, { width, 3 }
       
	THLongStorage* sizeStorage   = THLongStorage_newWithData(sizedata, 2);
	THLongStorage* strideStorage = THLongStorage_newWithData(stridedata, 2);
       
	if( !sizeStorage || !strideStorage )
	{
		printf("[rovernet]  failed to alloc size/stride storage\n");
		return NULL;
	}

	t->cpuTensor = THFloatTensor_new();

	if( !t->cpuTensor )
	{
		printf("[rovernet]  failed to create CPU THFloatTensor()\n");
		return NULL;
	}

	THFloatTensor_setStorage(t->cpuTensor, cpuStorage, 0LL, sizeStorage, strideStorage);

	
	// alloc GPU tensor
	THCudaStorage* gpuStorage = THCudaStorage_newWithData(THC, t->gpuPtr, elem);

	if( !gpuStorage )
	{
		printf("[rovernet]  failed to alloc GPU THCudaStorage\n");
		return NULL;
	}

	t->gpuTensor = THCudaTensor_new(THC);

	if( !t->cpuTensor )
	{
		printf("[rovernet]  failed to create GPU THCudaTensor()\n");
		return NULL;
	}

	THCudaTensor_setStorage(THC, t->gpuTensor, gpuStorage, 0LL, sizeStorage, strideStorage);


	// save variables
	t->width    = width;
	t->height   = height;
	t->depth    = depth;
	t->elements = elem;
	t->size     = size;

	printf("[rovernet]  allocated %u x %u x %u float tensor (%zu bytes)\n", width, height, depth, size);
	return t;
}


// init
bool roverNet::init()
{
	// create LUA environment
	L = luaL_newstate();

	if( !L )
	{
		printf("[rovernet]  failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("[rovernet]  created new lua_State\n");


	// load LUA libraries
	luaL_openlibs(L);
	printf("[rovernet]  opened LUA libraries\n");


	// load rovernet script
	printf("[rovernet]  loading '%s' \n", SCRIPT_FILENAME);
	const int res = luaL_dofile(L, SCRIPT_FILENAME);

	if( res == 1 ) 
	{
		printf("[rovernet]  error loading script: %s\n", SCRIPT_FILENAME);
		const char* luastr = lua_tostring(L,-1);

		if( luastr != NULL )
			printf("%s\n", luastr);
	}


	// get cuTorch state
	lua_getglobal(L, "cutorch");
	lua_getfield(L, -1, "_state");
	THC = (THCState*)lua_touserdata(L, -1);
	lua_pop(L, 2);

	if( !THC )
	{
		printf("[rovernet]  failed to retrieve cuTorch operating state\n");
		return false;
	}

	printf("[rovernet]  cuTorch numDevices:  %i\n", THC->numDevices);

	
	return true;
}



// updateNetwork
bool roverNet::updateNetwork( roverNet::Tensor* input, roverNet::Tensor* reward, roverNet::Tensor* output )
{
	lua_getglobal(L, SCRIPT_FUNC_NAME);

	if( input != NULL )
		luaT_pushudata(L, (void*)input->cpuTensor, "torch.FloatTensor");
		//luaT_pushudata(L, (void*)input->gpuTensor, "torch.CudaTensor");
		
	if( reward != NULL )
		luaT_pushudata(L, (void*)reward->cpuTensor, "torch.FloatTensor");

	if( output != NULL )
		luaT_pushudata(L, (void*)output->cpuTensor, "torch.FloatTensor");
	

	const int num_params = 3;
	const int num_result = 0;

	const int f_result = lua_pcall(L, num_params, num_result, 0);
	printf("[rovernet]  %s() ran (res=%i)\n", SCRIPT_FUNC_NAME, f_result);

	if( f_result != 0 )
	{
		printf("[rovernet]  error running %s   %s\n", SCRIPT_FUNC_NAME, lua_tostring(L, -1));
		return false;
	}

	return true;
}


