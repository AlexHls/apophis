cmake_minimum_required(VERSION 3.28)

project(apophis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src)

# Parthenon
add_subdirectory(submodules/parthenon)

# Singularity-EOS
add_subdirectory(submodules/singularity-eos)
