# Include test sources
file(GLOB TEST_SOURCES src/*.cpp src/*.cu)

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
        sf-nvcc-lib
        GTest::GTest
        GTest::Main
)

include(GoogleTest)
gtest_discover_tests(safecuda_tests)
