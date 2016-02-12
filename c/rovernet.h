/*
 * rovernet
 */

#ifndef __ROVERNET_H_
#define __ROVERNET_H_


#include <stdio.h>
#include <string.h>


extern "C" 
{ 
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}



/*
 * roverNet primary object
 */
class roverNet
{
public:
	static roverNet* Create();
	~roverNet();

private:
	roverNet();
	bool Init();

	lua_State* L;		/**< Lua/Torch7 operating environment */
};


#endif
