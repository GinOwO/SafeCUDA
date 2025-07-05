cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Set CUDA compiler before project declaration
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 86)

project(SafeCUDA LANGUAGES CXX CUDA)

# Modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA-specific settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(include)

# Debug/Release specific compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
    add_definitions(-DDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
else()
    message(STATUS "Building in default mode")
endif()

# Create the main SafeCUDA library
add_library(safecuda SHARED
    src/safecuda.cpp
    src/safecuda.cu
)

# Link CUDA libraries
target_link_libraries(safecuda 
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)

# Set properties for CUDA compilation
set_target_properties(safecuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler-specific flags
target_compile_options(safecuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Create test executable
add_executable(safecuda_test
    src/safecuda.cpp
    src/safecuda.cu
)

target_link_libraries(safecuda_test
    CUDA::cudart
    CUDA::cuda_driver
)

set_target_properties(safecuda_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(safecuda_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# # Installation rules
# install(TARGETS safecuda DESTINATION lib)
# install(FILES include/safecuda.h include/safecuda.cuh DESTINATION include)
