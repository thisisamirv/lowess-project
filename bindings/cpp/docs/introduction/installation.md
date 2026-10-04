\page installation Installation

# Installation

Install the LOWESS library for your preferred language.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64
```

## Pre-built Binaries (Linux (ARM64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-arm64.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-arm64
```

## Pre-built Binaries (Linux (x86), 32-bit)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x86.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
g++ -m32 -std=c++17 -I. -o myapp myapp.cpp -L. -lfastlowess-linux-x86
```

## Pre-built Binaries (Linux (ARMv7), hard-float)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-armv7.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
arm-linux-gnueabihf-g++ -std=c++17 -I. -o myapp myapp.cpp -L. -lfastlowess-linux-armv7
```

## Pre-built Binaries (Linux (x64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64-musl.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64-musl
```

## Pre-built Binaries (Linux (ARM64), musl/Alpine)

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-arm64-musl.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-arm64-musl
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-x64.dylib
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-x64
```

## Pre-built Binaries (macOS (ARM64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-arm64.dylib
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-arm64
```

## Pre-built Binaries (Android)

Choose the shared library matching the Android ABI used by your application:

| ABI | Release asset |
| --- | --- |
| `arm64-v8a` | `libfastlowess-android-arm64-v8a.so` |
| `armeabi-v7a` | `libfastlowess-android-armeabi-v7a.so` |
| `x86` | `libfastlowess-android-x86.so` |
| `x86_64` | `libfastlowess-android-x86_64.so` |

Download the C++ and C headers (`fastlowess.hpp`, `fastlowess.h`, and `fastlowess_version.h`) from the same release. Bundle and link the `.so` for the ABI being built by the Android NDK.

## Pre-built Binaries (iOS)

The release provides static archives for physical devices and simulators. Use only the archive matching the active Xcode destination:

| Destination | Rust target | Release asset |
| --- | --- | --- |
| iOS device (arm64) | `aarch64-apple-ios` | `libfastlowess-ios-arm64.a` |
| iOS simulator (Apple silicon) | `aarch64-apple-ios-sim` | `libfastlowess-ios-simulator-arm64.a` |
| iOS simulator (Intel) | `x86_64-apple-ios` | `libfastlowess-ios-simulator-x86_64.a` |

Download the C++ and C headers (`fastlowess.hpp`, `fastlowess.h`, and `fastlowess_version.h`) from the same release and link the matching static archive into your app or framework.

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-x64.dll
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-x64.lib
cl /std:c++17 myapp.cpp /link fastlowess-win32-x64.lib
```

## Pre-built Binaries (Windows (ARM64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-arm64.dll
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-arm64.lib
cl /std:c++17 myapp.cpp /link fastlowess-win32-arm64.lib
```

## Pre-built Binaries (Windows (x64), MinGW-w64)

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-x64-gnu.dll
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-win32-x64-gnu.dll.a
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.h
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess_version.h
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
