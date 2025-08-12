cmake_minimum_required(VERSION 3.18)

message(STATUS "Building sf-nvcc tool")

find_package(ZLIB REQUIRED)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
        COMMAND gzip -c ${CMAKE_CURRENT_SOURCE_DIR}/text/help.txt | xxd -i -n help_txt_gz > ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/text/help.txt
        COMMENT "Compressing and embedding help.txt"
)

add_library(sf-nvcc-lib SHARED
        src/nvcc_wrapper.cpp
        src/ptx_modifier.cpp
        src/pattern_matcher.cpp
        src/sf_options.cpp
        ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
)
target_include_directories(sf-nvcc-lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sf-nvcc-lib PRIVATE ZLIB::ZLIB)
set_target_properties(sf-nvcc-lib PROPERTIES CXX_STANDARD 23)

add_executable(sf-nvcc src/main.cpp)
target_include_directories(sf-nvcc PRIVATE include)
target_link_libraries(sf-nvcc PRIVATE sf-nvcc-lib)
set_target_properties(sf-nvcc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CXX_STANDARD 23
)

target_compile_definitions(sf-nvcc-lib PUBLIC
        PROJECT_VERSION="${PROJECT_VERSION}"
)
