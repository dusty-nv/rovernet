/*
 * rovernet
 */

#ifdef ROVERNET_CONSOLE

#include "rovernet.h"



bool init()
{
	lua_State* L = luaL_newstate();

	if( !L )
	{
		printf("failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("created new lua_State\n");

	luaL_openlibs(L);
	printf("opened lua libraries\n");
	
	printf("closing lua_State\n");
	lua_close(L);

	return true;
}


int main( int argc, char** argv )
{
	printf("rovernet-console\n\n");

	if( !init() )
	{
		printf("failed to init lua, exiting rovernet-console\n");
		return 0;
	}

	return 0;
}

#endif
