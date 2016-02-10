/*
 * rovernet
 */

#ifdef ROVERNET_CONSOLE

#include "rovernet.h"
#include <sys/stat.h>
#include <TH/THTensor.h>


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


extern "C" void lua_ffi_msg( const char* msg )
{
	printf("lua_ffi_msg(%s)\n", msg);
}


extern "C" bool fillTensor(int rows, int cols, THByteTensor* emptyTensor) 
{
    bool result = false;
    int  tLen   = rows * cols;
   
	printf("fillTensor(%i, %i)\n", rows, cols);

    unsigned char* theData = (unsigned char*) malloc(sizeof(unsigned char) * tLen);
    if(theData) {
        for(int i = 0; i < tLen; ++i)
            theData[i] = (unsigned char) i;
    }
   
    THByteStorage* theStorage = THByteStorage_newWithData(theData, tLen);
   
    if(theStorage) {
       
        long sizedata[2]   = { rows, cols };
        long stridedata[2] = { cols, 1};
       
        THLongStorage* size    = THLongStorage_newWithData(sizedata, 2);
        THLongStorage* stride  = THLongStorage_newWithData(stridedata, 2);
       
        THByteTensor_setStorage(emptyTensor, theStorage, 0LL, size, stride);
        result = true;
    }


    return result;
}



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

