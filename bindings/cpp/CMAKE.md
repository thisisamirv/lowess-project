# CMake Guide

This document covers CMake-based consumption of the C++ binding.

## Compiler Support

| Compiler | Platform | Status | CI job |
| --- | --- | --- | --- |
| GCC (`g++`) | Linux | Supported | `C++ (ubuntu-latest)` |
| Clang (`clang++`) | Linux | Supported | `C++ (ubuntu-latest, clang)` |
| Clang (AppleClang) | macOS | Supported | `C++ (macos-latest)` |
| MSVC (`cl.exe`) | Windows | Supported | `C++ (windows-latest)` |
| clang-cl | Windows | Supported | `C++ (windows-latest, clang-cl)` |
| MinGW-w64 (`g++`) | Windows | Experimental | `C++ (windows-latest, MinGW-w64)` |
| Intel oneAPI (`icpx`) | Linux | Experimental | `C++ (ubuntu-latest, Intel oneAPI)` |

"Experimental" means the compiler is exercised in CI on a best-effort, non-blocking basis (`continue-on-error: true`) and isn't a release gate yet. To build the CMake test harness with clang-cl instead of MSVC, pass the toolset to CMake: `cmake -T ClangCL ...` (or `make cpp-dev CPP_CMAKE_TOOLSET="-T ClangCL"` from `bindings/cpp/Makefile`).

## Package Config Support

The C++ binding generates and installs a standard CMake package config, so downstream projects can use `find_package(fastlowess CONFIG REQUIRED)` instead of wiring include directories and libraries manually.

## Windows Quick Start

Build and install the package:

```powershell
cmake -S bindings/cpp -B build-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp --config Release
cmake --install build-cpp --config Release --prefix "$env:LOCALAPPDATA/fastlowess"
```

Consumer project:

```cmake
find_package(fastlowess CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE fastlowess::fastlowess)
```

If the package was installed to a non-standard prefix, point `CMAKE_PREFIX_PATH` at that install root.

## Build Tree Discovery

If CMake package registry support is enabled, configuring `bindings/cpp` also registers the build tree automatically. That allows another local project to resolve:

```cmake
find_package(fastlowess CONFIG REQUIRED)
```

without manual include or library setup.

If your environment disables the CMake package registry, use an installed prefix and set `CMAKE_PREFIX_PATH`, or set `fastlowess_DIR` directly to the directory containing `fastlowessConfig.cmake`.
