# CMake Guide

This document covers CMake-based consumption of the C++ binding.

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
