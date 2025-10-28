cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SafeCUDAExamples VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(example
        example.cpp
        kernel.cu
)

target_compile_options(example PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--keep>
        $<$<COMPILE_LANGUAGE:CUDA>:--keep-dir=examples>
)

target_link_libraries(example
        safecuda
        CUDA::cudart
        CUDA::cuda_driver
        ${CMAKE_DL_LIBS}
)

set_target_properties(example PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)
