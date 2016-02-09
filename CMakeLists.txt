
cmake_minimum_required(VERSION 2.8)

project(rovernet)

set(ROVERNET_CONSOLE "ON" CACHE BOOL "If ON, build rovernet as a command-line console environment.  If OFF, build rovernet as a shared library.")
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


# build rovernet C/C++ interface
include_directories(${PROJECT_BINARY_DIR}/torch/include)
link_directories(${PROJECT_BINARY_DIR}/torch/lib)

file(GLOB rovernetSources c/*.cpp)
file(GLOB rovernetIncludes c/*.h)

if(ROVERNET_CONSOLE)
	message("Building rovernet utility, defining -DROVERNET_CONSOLE")
	add_definitions(-DROVERNET_CONSOLE )
	add_executable(rovernet-console ${rovernetSources})
	target_link_libraries(rovernet-console luajit)
else()
	message("building librovernet, defining -DROVERNET_LIB")
	add_definitions(-DROVERNET_LIB )
endif()


