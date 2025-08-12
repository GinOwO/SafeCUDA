# Include test sources
file(GLOB TEST_SOURCES src/*.cpp src/*.cu)

add_executable(safecuda_tests ${TEST_SOURCES})

target_include_directories(safecuda_tests PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(safecuda_tests
        safecuda
        GTest::GTest
        GTest::Main
)

include(GoogleTest)
gtest_discover_tests(safecuda_tests)
