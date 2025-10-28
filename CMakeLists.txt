cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)

set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 9)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

project(SafeCUDA VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
include_directories(include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -rdc=true")
    add_definitions(-DDEBUG)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building in Release mode")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -rdc=true")
    add_definitions(-DNDEBUG)
endif ()

option(BUILD_SAFECUDA_TOOLS "Build SafeCUDA tools (sf-nvcc)" ON)
option(BUILD_SAFECUDA_EXAMPLES "Build SafeCUDA examples" ON)

# Build libsafecuda ------------------------------------------------------------
add_library(safecuda SHARED
        src/safecuda.cpp
        src/safecache.cu
)

target_link_libraries(safecuda
        CUDA::cudart
        CUDA::cuda_driver
        ${CMAKE_DL_LIBS}
)

set_target_properties(safecuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

target_compile_options(safecuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_definitions(safecuda PRIVATE
        PROJECT_VERSION_MAJOR="${PROJECT_VERSION_MAJOR}"
        PROJECT_VERSION_MINOR="${PROJECT_VERSION_MINOR}"
        PROJECT_VERSION_PATCH="${PROJECT_VERSION_PATCH}"
        PROJECT_VERSION="${PROJECT_VERSION}"
)

# Tools ------------------------------------------------------------------------
if (BUILD_SAFECUDA_TOOLS)
    message(STATUS "Building SafeCUDA tools")
    add_subdirectory(tools)
endif ()

# Examples ---------------------------------------------------------------------
if (BUILD_SAFECUDA_EXAMPLES)
    message(STATUS "Building Examples")
    add_subdirectory(examples)
endif ()

# Tests ------------------------------------------------------------------------
find_package(GTest REQUIRED)
enable_testing()

# Add tests subdirectory
add_subdirectory(tests)

# # Installation rules
# install(TARGETS safecuda DESTINATION lib)
# install(FILES include/safecuda.h include/safecuda.cuh DESTINATION include)
# if(BUILD_SAFECUDA_TOOLS)
#     install(TARGETS sf-nvcc DESTINATION bin)
# endif()
