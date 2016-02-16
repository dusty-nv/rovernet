/*
 * rovernet
 */

#include "rovernet.h"


#define SCRIPT_FILENAME  "rovernet.lua"
#define SCRIPT_FUNC_NAME "update_network"



// constructor
roverNet::roverNet()
{
	L = NULL;
}


// destructor
roverNet::~roverNet()
{

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


// init
bool roverNet::init()
{
	L = luaL_newstate();

	if( !L )
	{
		printf("[roverNet]  failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("[roverNet]  created new lua_State\n");

	luaL_openlibs(L);
	printf("[roverNet]  opened lua libraries\n");

	
	return true;
}


// setImageSize
void roverNet::setImageSize( size_t width, size_t height, size_t pitch )
{

}


// updateNetwork
bool roverNet::updateNetwork( float* image, float* reward, float* output )
{
	lua_getglobal(L, SCRIPT_FUNC_NAME);

	/*luaT_pushudata(L, (void*)tensor, "torch.FloatTensor");*/

	const int num_params = 1;
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

