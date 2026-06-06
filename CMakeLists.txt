cmake_minimum_required(VERSION 3.24)

project(apophis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src)

option(Apophis_ENABLE_TESTING "Enable testing" ON)
option(Apophis_ENABLE_CUDA "Enable cuda for Apophis and all dependencies" OFF)
option(Apophis_ENABLE_HDF5 "Enable HDF5 for Apophis and all dependencies" ON)
option(Apophis_ENABLE_MPI "Enable MPI for Apophis and all dependencies" ON)
option(Apophis_ENABLE_OPENMP "Enable OpenMP for Apophis and parthenon" OFF)

set(PARTHENON_ENABLE_PYTHON_MODULE_CHECK ${Apophis_ENABLE_TESTING} CACHE BOOL "Check if local python version contains all modules required for running tests.")

set(PARTHENON_ENABLE_TESTING OFF CACHE BOOL "Disable Parthenon testing.")
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "Enable Parthenon examples.")

# If the user doesn't specify a build type, prefer RelWithDebInfo
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
    STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

# CUDA
if(Apophis_ENABLE_CUDA)
  set(SINGULARITY_USE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "" FORCE)
  set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON CACHE BOOL "" FORCE)
endif()

if(Apophis_ENABLE_MPI)
  message("Enabling MPI for Apophis and all dependencies")
  find_package(MPI REQUIRED)
else()
  set(PARTHENON_DISABLE_MPI ON CACHE BOOL "" FORCE)
endif()

# Parthenon
add_subdirectory(submodules/parthenon)

# Singularity-EOS
# singularity common
message("Configuring singularity")
set(SINGULARITY_USE_KOKKOS ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_FORTRAN OFF CACHE BOOL "" FORCE)
set(SINGULARITY_FORCE_SUBMODULE_MODE ON CACHE BOOL "" FORCE)
set(SINGULARITY_USE_KOKKOSKERNELS OFF CACHE BOOL "" FORCE)
set(SINGULARITY_BUILD_CLOSURE OFF CACHE BOOL "" FORCE)
set(SINGULARITY_USE_SPINER ON CACHE BOOL "Use Spiner for Singularity-EOS")
set(SINGULARITY_USE_HELMHOLTZ ON CACHE BOOL "Use Helmholtz EOS for Singularity-EOS")

# Kill cmake's package registry because it can interfere
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON CACHE BOOL "" FORCE)

add_subdirectory(submodules/singularity-eos)

if (Apophis_ENABLE_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(tst/regression)
endif()
