# rovernet

rovernet is a package that uses deep reinforcement learning to teach self-driving robots from experience how to operate safely in unstructured environments.
rovernet is an end-to-end deep learning system — by recieving the robot's stereo field and directly controlling motor outputs, it's able to intuitively sense and avoid obstacles and obstructions in it's environment.

rovernet's learning-driven navigation framework provides a building block upon which higher-level autonomous functions can safely be layered.
The CNN/RNN's that comprise the network and it's associated action/state reward functions are implemented in Lua using Torch7.
A convenient C-library interface is provided for integrating rovernet with low-level sensors and robotic control components.  

A self-contained sandbox holds the Torch/Lua components locally for improved code management & configuration control.

Underneath, rovernet and Torch are optimized for GPU acceleration with CUDA, and are ideal for deploying onboard embedded platforms with NVIDIA's Jetson device.


# Building from Source

First, to obtain the rovernet sources, clone the repository from github:

`git clone http://github.org/dusty-nv/rovernet`

Next, configure and build the project.  Prerequisites will automatically be installed and built into the self-contained sandbox environment, including Lua, Torch, and it's dependencies.
Lua, Torch, and OpenBLAS are placed in a local build directory specified during configuration.  The sandbox improves packaging and allows managing multiple concurrent branches on a system without dependency conflicts. 

To compile rovernet, run these commands from terminal:

    cd rovernet
    mkdir build
    cd build
    cmake ../
    make


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
