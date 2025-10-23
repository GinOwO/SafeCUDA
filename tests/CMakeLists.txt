# Include test sources
set(TEST_SOURCES
        src/bootstrap_test.cpp
        src/test_cuda_linking.cpp
        src/test_cuda_linking.cu
        src/test_safecuda_runtime.cpp
        src/test_safecuda_runtime.cu
        src/test_sf_nvcc_parsing.cpp
        src/test_sf_nvcc_pattern_matching.cpp
        src/test_sf_nvcc_ptx_injector.cpp
)

add_executable(safecuda_tests ${TEST_SOURCES})

SET(TOOL_HEADERS
        ${PROJECT_SOURCE_DIR}/tools/sf-nvcc/include
)

target_include_directories(safecuda_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TOOL_HEADERS}
)

target_link_libraries(safecuda_tests
        safecuda
        safecuda_device
        sf-nvcc-lib
        GTest::GTest
        GTest::Main
)

target_compile_definitions(safecuda_tests PRIVATE
        PROJECT_VERSION_MAJOR="${PROJECT_VERSION_MAJOR}"
        PROJECT_VERSION_MINOR="${PROJECT_VERSION_MINOR}"
        PROJECT_VERSION_PATCH="${PROJECT_VERSION_PATCH}"
        PROJECT_VERSION="${PROJECT_VERSION}"
)


include(GoogleTest)
gtest_discover_tests(safecuda_tests)
