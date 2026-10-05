cmake_minimum_required(VERSION 3.24)
project(fastlowess-vcpkg-consumer LANGUAGES CXX)

find_package(fastlowess CONFIG REQUIRED)
add_executable(fastlowess-consumer main.cpp)
target_link_libraries(fastlowess-consumer PRIVATE fastlowess::fastlowess)
add_custom_command(
    TARGET fastlowess-consumer
    POST_BUILD
    COMMAND
        "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:fastlowess::fastlowess>"
        "$<TARGET_FILE_DIR:fastlowess-consumer>"
)
enable_testing()
add_test(NAME fastlowess-consumer COMMAND fastlowess-consumer)
