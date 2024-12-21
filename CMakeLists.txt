cmake_minimum_required(VERSION 3.28)

project(apophis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src)

option(Apophis_ENABLE_TESTING "Enable testing" ON)
set(PARTHENON_ENABLE_PYTHON_MODULE_CHECK ${Apophis_ENABLE_TESTING} CACHE BOOL "Check if local python version contains all modules required for running tests.")

set(PARTHENON_ENABLE_TESTING OFF CACHE BOOL "Disable Parthenon testing.")

# Parthenon
add_subdirectory(submodules/parthenon)

# Singularity-EOS
add_subdirectory(submodules/singularity-eos)
set(SINGULARITY_USE_SPINER ON CACHE BOOL "Use Spiner for Singularity-EOS")
set(SINGULARITY_USE_HELMHOLTZ ON CACHE BOOL "Use Helmholtz EOS for Singularity-EOS")

if (Apophis_ENABLE_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(tst/regression)
endif()
