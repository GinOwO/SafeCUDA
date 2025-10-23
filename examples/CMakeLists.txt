cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(SafeCUDAExamples VERSION ${PROJECT_VERSION} LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

#find_program(SF_NVCC_EXECUTABLE
#        NAMES sf-nvcc
#        PATHS
#        ${CMAKE_BINARY_DIR}/tools/sf-nvcc
#        ${CMAKE_SOURCE_DIR}/build/tools/sf-nvcc
#        /usr/local/bin
#        /opt/safecuda/bin
#        DOC "SafeCUDA NVCC wrapper executable"
#)
#
#if (SF_NVCC_EXECUTABLE)
#    set(CMAKE_CUDA_COMPILER ${SF_NVCC_EXECUTABLE})
#    message(STATUS "Using SafeCUDA compiler: ${SF_NVCC_EXECUTABLE}")
#    set(USING_SAFECUDA TRUE)
#else ()
#    find_program(NVCC_EXECUTABLE
#            NAMES nvcc
#            PATHS /usr/local/cuda/bin
#            DOC "NVIDIA CUDA compiler"
#    )
#
#    if (NVCC_EXECUTABLE)
#        set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
#        message(STATUS "SafeCUDA not found, using standard NVCC: ${NVCC_EXECUTABLE}")
#        set(USING_SAFECUDA FALSE)
#    else ()
#        message(FATAL_ERROR "Neither sf-nvcc nor nvcc found. Install CUDA toolkit and SafeCUDA.")
#    endif ()
#endif ()

add_executable(example
        example.cpp
        kernel1.cu
        kernel2.cu
        montecarlo.cu
        out_of_bounds.cu
)

#target_compile_options(example PRIVATE
#        $<$<COMPILE_LANGUAGE:CUDA>:-sf-cache-size=1024>
#        $<$<COMPILE_LANGUAGE:CUDA>:-sf-bounds-check>
#        $<$<COMPILE_LANGUAGE:CUDA>:-sf-debug>
#)

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
