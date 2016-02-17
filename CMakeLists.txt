
cmake_minimum_required(VERSION 2.8)

project(rovernet)

set(ROVERNET_CONSOLE "ON" CACHE BOOL "If ON, build rovernet as a command-line console environment.")
set(ROVERNET_LIBRARY "ON" CACHE BOOL "If ON, build librovernet shared library.")
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_OPENBLAS "YES" CACHE BOOL "If YES, will download & build OpenBLAS (Torch backend) into sandbox")


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_OPENBLAS}
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("Finished installing dependencies")
endif()


find_package(CUDA)

# Pass options to NVCC
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3 -gencode arch=compute_53,code=sm_53
)


# build rovernet C/C++ interface
include_directories(${PROJECT_BINARY_DIR}/torch/include ${PROJECT_BINARY_DIR}/torch/include/TH)
link_directories(${PROJECT_BINARY_DIR}/torch/lib)

file(GLOB rovernetSources c/*.cpp cuda/*.cu)
file(GLOB rovernetIncludes c/*.h cuda/*.h)

if(ROVERNET_CONSOLE)
	message("Building rovernet utility, defining -DROVERNET_CONSOLE")
	add_definitions(-DROVERNET_CONSOLE)
	cuda_add_executable(rovernet-console ${rovernetSources})
	target_link_libraries(rovernet-console luajit luaT TH THC)
endif()

if(ROVERNET_LIBRARY)
	message("Building librovernet, defining -DROVERNET_LIB")
	add_definitions(-DROVERNET_LIB)
	cuda_add_library(rovernet SHARED ${rovernetSources})
	target_link_libraries(rovernet luajit luaT TH THC)
endif()


# transfer all LUA scripts to bin directory
file(GLOB luaScripts lua/*.lua)

foreach(luaScript ${luaScripts})
	message("-- Configuring ${luaScript}")
	configure_file(${luaScript} ${PROJECT_BINARY_DIR} COPYONLY)
endforeach()

