/*
 * rovernet
 */

#include "rovernet.h"



// constructor
roverNet::roverNet()
{
	L = NULL;
}


// destructor
roverNet::~roverNet()
{

}



// Init
bool roverNet::Init()
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
