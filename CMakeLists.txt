
cmake_minimum_required(VERSION 2.8)

project(rovernet)

set(LUA_ROOT "" CACHE FILEPATH "Lua/Torch root path (containing include, lib)")
set(ROVERNET_CONSOLE "OFF" CACHE BOOL "If ON, build rovernet as a command-line console environment.  If OFF, build rovernet as a shared library.")

if( LUA_ROOT STREQUAL "" )
	message("Lua:        LUA_ROOT path not set")
	message("Lua:        use cmake -DLUA_ROOT=<path> to set Lua/Torch root path")
	return()
endif()

message("Lua/Torch include path:  ${LUA_ROOT}/include")
include_directories(${LUA_ROOT}/include)
link_directories(${LUA_ROOT}/lib)

file(GLOB rovernetSources c/*.cpp)
file(GLOB rovernetIncludes c/*.h)

if(ROVERNET_LIB)

message("building librovernet, defining -DROVERNET_LIB")
add_definitions(-DROVERNET_LIB )

else()

message("building rovernet utility, defining -DROVERNET_CONSOLE")
add_definitions(-DROVERNET_CONSOLE )
add_executable(rovernet-console ${rovernetSources})
target_link_libraries(rovernet-console luajit)

endif()


