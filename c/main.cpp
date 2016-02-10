/*
 * rovernet
 */

#ifdef ROVERNET_CONSOLE

#include "rovernet.h"
#include <sys/stat.h>



#define SCRIPT_FILENAME "main.lua"
#define LUA_FUNC_NAME "ex_lua_func"

void call_lua( lua_State* L )
{
	lua_getglobal(L, LUA_FUNC_NAME);
	lua_pushnumber(L, 1.0);
	lua_pushnumber(L, 2.0);

	// call lua func (2 arguments, 1 result)
	const int f_result = lua_pcall(L, 2, 1, 0);

	printf("%s() ran (res=%i)\n", LUA_FUNC_NAME, f_result);
	
	if( f_result != 0 )
	{
		printf("error running %s()  %s\n", LUA_FUNC_NAME, lua_tostring(L, -1));
		return;
	}

	// return value
	const double ret = lua_tonumber(L, -1);
	lua_pop(L, 1);
	printf("%s() => %lf\n", LUA_FUNC_NAME, ret);
}


int lua_c_square( lua_State* L )
{
	double d = lua_tonumber(L, 1);	// get argument
	lua_pushnumber(L, d * d );
	return 1;
};


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
	

	lua_pushcfunction(L, lua_c_square);
	lua_setglobal(L, "my_square");


 	// load and run file
	const int res = luaL_dofile(L, SCRIPT_FILENAME);

	if( res == 1 ) 
	{
		printf("Error executing resource: %s\n", SCRIPT_FILENAME);
		const char* luastr = lua_tostring(L,-1);

		if( luastr != NULL )
			printf("%s\n", luastr);
	}

	call_lua(L);

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

