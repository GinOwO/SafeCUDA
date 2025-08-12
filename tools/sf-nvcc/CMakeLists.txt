cmake_minimum_required(VERSION 3.18)

message(STATUS "Building sf-nvcc tool")

find_package(ZLIB REQUIRED)

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
        COMMAND gzip -c ${CMAKE_CURRENT_SOURCE_DIR}/text/help.txt | xxd -i -n help_txt_gz > ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/text/help.txt
        COMMENT "Compressing and embedding help.txt"
)

SET(CPP_SOURCES
        src/main.cpp
        src/nvcc_wrapper.cpp
        src/ptx_modifier.cpp
        src/pattern_matcher.cpp
        src/sf_options.cpp
)

add_executable(sf-nvcc
        ${CPP_SOURCES}
        ${CMAKE_BINARY_DIR}/help_text_compressed.cpp
)

target_include_directories(sf-nvcc PRIVATE include)

target_link_libraries(sf-nvcc ZLIB::ZLIB)

set_target_properties(sf-nvcc PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CXX_STANDARD 23
)

target_compile_definitions(sf-nvcc PRIVATE
        PROJECT_VERSION="${PROJECT_VERSION}"
)
