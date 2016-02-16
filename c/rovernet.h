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

	void setImageSize( size_t width, size_t height, size_t pitch );
	bool updateNetwork( float* image, float* reward, float* output );

private:
	roverNet();
	bool init();

	lua_State* L;		/**< Lua/Torch7 operating environment */
};


#endif
