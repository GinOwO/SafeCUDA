cmake_minimum_required(VERSION 3.18)

message(STATUS "Building sf-nvcc tool")

add_executable(sf-nvcc
        src/main.cpp
        src/nvcc_wrapper.cpp
        src/ptx_modifier.cpp
        src/pattern_matcher.cpp
)

target_include_directories(sf-nvcc PRIVATE include)

set_target_properties(sf-nvcc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CXX_STANDARD 17
)
