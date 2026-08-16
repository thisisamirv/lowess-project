cmake_minimum_required(VERSION 3.12)
project(fastlowess_tests CXX)

set(CMAKE_CXX_STANDARD 17)

# Paths
set(BINDINGS_DIR "${CMAKE_SOURCE_DIR}/../../bindings/cpp")
include_directories("${BINDINGS_DIR}/include")

set(FASTLOWESS_LIB
    ""
    CACHE FILEPATH
    "Path to the compiled fastlowess C++ library"
)
set(FASTLOWESS_LIB_DIR
    "${CMAKE_SOURCE_DIR}/../../target/release"
    CACHE PATH
    "Directory containing the compiled fastlowess C++ library"
)

if(FASTLOWESS_LIB)
    get_filename_component(FASTLOWESS_LIB "${FASTLOWESS_LIB}" ABSOLUTE)
else()
    if(WIN32)
        if(MSVC)
            set(FASTLOWESS_CANDIDATE_NAMES
                fastlowess_cpp
                fastlowess_cpp.dll.lib
            )
        else()
            set(FASTLOWESS_CANDIDATE_NAMES
                fastlowess_cpp
                libfastlowess_cpp.dll.a
                libfastlowess_cpp.a
            )
        endif()
    else()
        set(FASTLOWESS_CANDIDATE_NAMES
            fastlowess_cpp
            libfastlowess_cpp.so
            libfastlowess_cpp.a
        )
    endif()

    find_library(
        FASTLOWESS_LIB_FOUND
        NAMES ${FASTLOWESS_CANDIDATE_NAMES}
        PATHS "${BINDINGS_DIR}/build" "${FASTLOWESS_LIB_DIR}"
        NO_DEFAULT_PATH
    )

    set(FASTLOWESS_LIB "${FASTLOWESS_LIB_FOUND}")
endif()

if(NOT FASTLOWESS_LIB OR FASTLOWESS_LIB MATCHES "-NOTFOUND$")
    message(
        FATAL_ERROR
        "Could not find libfastlowess_cpp. Please build bindings/cpp first."
    )
endif()

add_executable(test_fastlowess_suite test_fastlowess.cpp)
if(WIN32)
    if(MSVC)
        # MSVC compiler - use Windows system libraries
        target_link_libraries(
            test_fastlowess_suite
            ${FASTLOWESS_LIB}
            ntdll
            ws2_32
            userenv
            bcrypt
        )
    else()
        # MinGW compiler - link against the GNU-compatible Rust artifacts.
        target_link_libraries(
            test_fastlowess_suite
            ${FASTLOWESS_LIB}
            ntdll
            ws2_32
            userenv
            bcrypt
            pthread
        )
    endif()
else()
    target_link_libraries(test_fastlowess_suite ${FASTLOWESS_LIB} pthread dl)
endif()

# Add test
enable_testing()
add_test(NAME test_fastlowess_suite COMMAND test_fastlowess_suite)
