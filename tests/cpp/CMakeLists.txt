cmake_minimum_required(VERSION 3.12)
project(fastlowess_tests CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/../../bindings/cpp")
include_directories("${BINDINGS_DIR}/include")

# Find library (built by bindings/cpp/CMakeLists.txt)
# We assume the library is in bindings/cpp/build
find_library(FASTLOWESS_LIB NAMES fastlowess_cpp PATHS "${BINDINGS_DIR}/build" "${CMAKE_SOURCE_DIR}/../../target/release" NO_DEFAULT_PATH)

if (NOT FASTLOWESS_LIB)
    message(FATAL_ERROR "Could not find libfastlowess_cpp. Please build bindings/cpp first.")
endif()

add_executable(test_fastlowess_suite test_fastlowess.cpp)
target_link_libraries(test_fastlowess_suite ${FASTLOWESS_LIB} pthread dl)

# Add test
enable_testing()
add_test(NAME test_fastlowess_suite COMMAND test_fastlowess_suite)
