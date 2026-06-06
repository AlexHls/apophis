# apophis

## Build

```bash
git clone https://github.com/AlexHls/apophis
cd apophis
mkdir build && cd build
cmake ..
make
```

### CUDA
To build with CUDA support, use the following command:

```bash
cmake -DApophis_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=${PWD}/../submodules/singularity-eos/utils/kokkos/bin/nvcc_wrapper ..
```

You **must** set the `CMAKE_CXX_COMPILER` variable to the path of the `nvcc_wrapper` script. The `singularity-eos` repository contains one such script, but you can point to any other script that sets the necessary environment variables for the CUDA compiler.

### CUDA on Arch Linux
If you are using Arch Linux, you can use the following command to build with CUDA support:

```bash
cmake -DApophis_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=${PWD}/../bin/nvcc_wrapper_archlinux ..
```

This command sets the `c++` compiler version to `g++-13` since the latest version (version) does not work properly with the CUDA compiler.
To use this wrapper, you need to have the `gcc13` package installed on your system.

## Run

Example:

```bash
./bin/apophis -i ../inputs/sod.in
```

## Test

```bash
ctest
```
