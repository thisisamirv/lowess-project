\page installation Installation

# Installation

Install the LOWESS library for your preferred language.

Each prebuilt platform archive contains that platform's library and the matching C++ and C headers. Download and extract the archive for your target; its files are placed in the current directory.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64.tar
tar -xf libfastlowess-linux-x64.tar
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64
```

## Pre-built Binaries (Linux (ARM64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-arm64.tar
tar -xf libfastlowess-linux-arm64.tar
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-arm64
```

## Pre-built Binaries (Linux (x86), 32-bit)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x86.tar
tar -xf libfastlowess-linux-x86.tar
g++ -m32 -std=c++17 -I. -o myapp myapp.cpp -L. -lfastlowess-linux-x86
```

## Pre-built Binaries (Linux (ARMv7), hard-float)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-armv7.tar
tar -xf libfastlowess-linux-armv7.tar
arm-linux-gnueabihf-g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastlowess-linux-armv7
```

## Pre-built Binaries (Linux (x64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64-musl.tar
tar -xf libfastlowess-linux-x64-musl.tar
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64-musl
```

## Pre-built Binaries (Linux (ARM64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-arm64-musl.tar
tar -xf libfastlowess-linux-arm64-musl.tar
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-arm64-musl
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-x64.tar
tar -xf libfastlowess-macos-x64.tar
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-x64
```

## Pre-built Binaries (macOS (ARM64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-arm64.tar
tar -xf libfastlowess-macos-arm64.tar
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-arm64
```

## Pre-built Binaries (Android)

Choose the shared library matching the Android ABI used by your application:

| ABI | Release archive |
| --- | --- |
| `arm64-v8a` | `libfastlowess-android-arm64-v8a.tar` |
| `armeabi-v7a` | `libfastlowess-android-armeabi-v7a.tar` |
| `x86` | `libfastlowess-android-x86.tar` |
| `x86_64` | `libfastlowess-android-x86_64.tar` |

Download and extract the archive for the ABI being built by the Android NDK. It contains the `.so` and all three headers.

## Pre-built Binaries (iOS)

The release provides static archives for physical devices and simulators. Use only the archive matching the active Xcode destination:

| Destination | Rust target | Release archive |
| --- | --- | --- |
| iOS device (arm64) | `aarch64-apple-ios` | `libfastlowess-ios-arm64.tar` |
| iOS simulator (Apple silicon) | `aarch64-apple-ios-sim` | `libfastlowess-ios-simulator-arm64.tar` |
| iOS simulator (Intel) | `x86_64-apple-ios` | `libfastlowess-ios-simulator-x86_64.tar` |

Download and extract the archive matching the active Xcode destination. It contains the static library and all three headers.

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-windows-x64-msvc.tar
tar -xf libfastlowess-windows-x64-msvc.tar
cl /std:c++17 myapp.cpp /link fastlowess-win32-x64.lib
```

## Pre-built Binaries (Windows (ARM64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-windows-arm64.tar
tar -xf libfastlowess-windows-arm64.tar
cl /std:c++17 myapp.cpp /link fastlowess-win32-arm64.lib
```

## Pre-built Binaries (Windows (x64), MinGW-w64)

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-windows-x64-gnu.tar
tar -xf libfastlowess-windows-x64-gnu.tar
g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastlowess-win32-x64-gnu
```

## From Source

```bash
# Install Rust first: https://rustup.rs/
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/cpp

# Build the library
cargo build --release

# Headers are at: include/fastlowess.hpp (C++)
# Library is at: target/release/libfastlowess_cpp.so (Linux)
#                target/release/libfastlowess_cpp.dylib (macOS)
#                target/release/fastlowess_cpp.dll (Windows)
```

## From conda-forge

```bash
conda install -c conda-forge libfastlowess
```

## From Spack

```bash
spack install fastlowess-cpp
```

The recipe links its homepage to the C++ documentation and checks that the installed headers and library directory exist. Recipes with the standalone smoke test can compile and run a small linear fit against the installed library:

```bash
spack test run --alias fastlowess-cpp-smoke fastlowess-cpp
spack test results -l fastlowess-cpp-smoke
```

## From vcpkg (overlay)

A prebuilt overlay is available for shared CPU libraries on Windows x64/ARM64, Linux x64/ARM64, and macOS x64/ARM64. Windows uses the dynamic MSVC runtime. Linux triplets containing `musl` select musl artifacts; other Linux triplets select glibc artifacts. Use dynamic-linkage triplets: the standard Linux and macOS triplets that default to static linkage are not supported. Rust and Cargo are not required, and Debug and Release consumers use the same upstream Release library. This port is not yet in vcpkg's curated registry.

From the repository root, run the command matching your native build host:

```sh
# Windows
vcpkg install fastlowess:x64-windows --overlay-ports=bindings/cpp/vcpkg
# Linux (using the dynamic community triplet)
vcpkg install fastlowess:x64-linux-dynamic --overlay-ports=bindings/cpp/vcpkg
# macOS (using the dynamic community triplet)
vcpkg install fastlowess:arm64-osx-dynamic --overlay-ports=bindings/cpp/vcpkg
```

The installed package provides `fastlowess::fastlowess` through `find_package(fastlowess CONFIG REQUIRED)`. See the [overlay packaging guide](https://github.com/thisisamirv/lowess-project/tree/main/bindings/cpp/vcpkg) for bootstrap commands, Debug/Release consumer tests, and submission steps.

---

## Verify Installation

```cpp
#include <fastlowess.hpp>
#include <iostream>
#include <vector>

int main() {
std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

fastlowess::Lowess model;
model.fit(x, y).value();

std::cout << "Installed successfully!" << std::endl;
return 0;
}
```

```output
Installed successfully!
```

## Check the Header and Library Versions

Cargo and CMake generate `fastlowess_version.h` from package metadata. Download it and `fastlowess.h` alongside `fastlowess.hpp` when using prebuilt binaries. The version header can be included on its own for compile-time checks, without linking the native library:

```cpp
#include <fastlowess_version.h>

static_assert(FASTLOWESS_CPP_VERSION_MAJOR >= 4,
     "This application requires fastlowess-cpp 4 or later");

int main() {}
```

The macros `FASTLOWESS_CPP_VERSION_MAJOR`, `FASTLOWESS_CPP_VERSION_MINOR`, `FASTLOWESS_CPP_VERSION_PATCH`, and `FASTLOWESS_CPP_VERSION_STRING` describe the headers used to compile your application. `fastlowess.hpp` includes this header automatically.

Use `cpp_version()` to identify the native library loaded at runtime:

```cpp
#include <fastlowess.hpp>
#include <iostream>

int main() {
 std::cout << "Header version: " << FASTLOWESS_CPP_VERSION_STRING << '\n';
 std::cout << "Loaded library version: " << cpp_version() << '\n';
}
```

```output
Header version: 4.1.0
Loaded library version: 4.1.0
```
