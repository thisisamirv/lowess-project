\page installation Installation

# Installation

Install the LOWESS library for your preferred language.

## Pre-built Binaries (Linux (x64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-x64.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-x64
```

## Pre-built Binaries (Linux (ARM64))

```bash
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-linux-arm64.so
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
g++ -o myapp myapp.cpp -L. -lfastlowess-linux-arm64
```

## Pre-built Binaries (macOS (x64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-x64.dylib
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-x64
```

## Pre-built Binaries (macOS (ARM64))

```bash
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/libfastlowess-macos-arm64.dylib
curl -LO https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
clang++ -o myapp myapp.cpp -L. -lfastlowess-macos-arm64
```

## Pre-built Binaries (Windows (x64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-x64.dll
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
cl myapp.cpp /link fastlowess-win32-x64.lib
```

## Pre-built Binaries (Windows (ARM64))

```powershell
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess-win32-arm64.dll
wget https://github.com/thisisamirv/lowess-project/releases/latest/download/fastlowess.hpp
cl myapp.cpp /link fastlowess-win32-arm64.lib
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
