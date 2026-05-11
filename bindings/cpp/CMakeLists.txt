cmake_minimum_required(VERSION 3.15)
project(fastlowess-cpp VERSION 1.3.0 LANGUAGES CXX)

include(FetchContent)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Import Corrosion for Rust integration
fetchcontent_declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.1
)
fetchcontent_makeavailable(Corrosion)

# Import the Rust crate
corrosion_import_crate(MANIFEST_PATH Cargo.toml)

# The target name is derived from [package] name in Cargo.toml ("fastlowess-cpp")

set_target_properties(fastlowess_cpp PROPERTIES EXPORT_NAME fastlowess)

# Configure include directories
# This ensures that #include "fastlowess.hpp" works for consumers
target_include_directories(fastlowess_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Create an alias for idiomatic usage (fastlowess::fastlowess)
add_library(fastlowess::fastlowess ALIAS fastlowess_cpp)

# Installation configuration
set(fastlowess_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/fastlowess")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/fastlowessConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/fastlowessConfig.cmake"
    INSTALL_DESTINATION "${fastlowess_install_cmakedir}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fastlowessConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(TARGETS fastlowess_cpp
    EXPORT fastlowessTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

export(EXPORT fastlowessTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/fastlowessTargets.cmake"
    NAMESPACE fastlowess::
)

install(EXPORT fastlowessTargets
    FILE fastlowessTargets.cmake
    NAMESPACE fastlowess::
    DESTINATION "${fastlowess_install_cmakedir}"
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/fastlowessConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/fastlowessConfigVersion.cmake"
    DESTINATION "${fastlowess_install_cmakedir}"
)

set(CMAKE_EXPORT_PACKAGE_REGISTRY ON)
export(PACKAGE fastlowess)

# Tests
set(fastlowess_test_source "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cpp/test_fastlowess.cpp")
set(fastlowess_test_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/cpp")

if(EXISTS "${fastlowess_test_source}")
    enable_testing()
    add_executable(test_fastlowess "${fastlowess_test_source}")
    target_link_libraries(test_fastlowess PRIVATE fastlowess_cpp)
    target_include_directories(test_fastlowess PRIVATE "${fastlowess_test_include_dir}")
    add_test(NAME test_fastlowess COMMAND test_fastlowess)
endif()
