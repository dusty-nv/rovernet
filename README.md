# rovernet

rovernet is a package that uses Deep Q-Learning to teach a robot how to drive itself in unstructured environments.
The CNN/RNN's that comprise the network and it's associated action/state reward functions are implemented in Lua using Torch7.
A C-library interface is provided for integrating rovernet with low-level sensors and robot control components.  
A self-contained sandbox environment holds the Lua/Torch7 components in a local directory for easy management & configuration control.


# Building from Source

First, to obtain the rovernet sources, clone the repository from github:

`git clone http://github.org/dusty-nv/rovernet`

Next, configure and build the project.  Prerequisites will automatically be installed and built into the self-contained sandbox, including Lua, Torch, and it's dependencies.
Lua, Torch, and OpenBLAS are placed locally in the build directory specified during configuration.  The sandbox improves packaging and allows managing multiple concurrent branches on a system without dependency conflicts. 

To compile rovernet, run these commands from terminal:

```> cd rovernet
> mkdir build
> cd build
> cmake ../
> make```


## Build Options

Dependencies will automatically be built the first time cmake is run, and skipped upon further invocations of cmake.
To manually force or re-trigger dependency installation, run cmake with the `BUILD_DEPS` flag enabled:

`> cmake ../ -DBUILD_DEPS=y -DBUILD_OPENBLAS=y`

The `BUILD_OPENBLAS` argument may be omitted if no change is required (it's a cached variable).
To manually skip the installation of Lua, Torch, and/or OpenBLAS while configuring (if for example, you already have those installed), turn off the cmake `BUILD_` flags:

`> cmake ../ -DBUILD_DEPS=n -DBUILD_OPENBLAS=n`

After initially disabling `BUILD_DEPS`, future invocations of cmake will continue to skip the dependencies, until it is re-enabled with `-DBUILD_DEPS=y` by the user.

## Cleaning the Sandbox

To remove the rovernet branch and Lua/Torch from the system, remove the build directory specified above during configuration:

`> rm -rf build`

The rovernet binaries and all the Lua/Torch components will be deleted, leaving the system in a pristine state.


# Project Directories

Underlying implementations in subdirectories:

- c c/c++ library interface to lua/torch environment
- lua torch scripts implementing DQ learner
- ros ROS node to c/c++ library
