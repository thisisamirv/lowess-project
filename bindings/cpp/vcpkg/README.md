# LOWESS for vcpkg

The `fastlowess` overlay port provides Rust-backed C++17 LOWESS smoothing with batch streaming, and online adapters. It builds the published `v4.1.0` source release and exposes the CMake target `fastlowess::fastlowess`.

This port is available as a repository overlay. It is not yet part of vcpkg's curated registry.

## Supported Configuration

| Setting | Support |
| --- | --- |
| Build host | Windows |
| Target triplet | `x64-windows` (MSVC) |
| Library linkage | Shared DLL |
| MSVC runtime | Dynamic CRT |
| Backend | CPU |
| Configurations | Debug and Release |

Static libraries, static CRT, ARM, MinGW, UWP, GPU builds, and non-Windows targets are not supported by this port. The manifest excludes unvalidated triplets.

## Requirements

- Visual Studio or Build Tools with the Desktop development with C++ workload and a Windows SDK.
- Rust 1.89 or newer, with Cargo and the `x86_64-pc-windows-msvc` target.
- An existing [vcpkg installation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging), with `VCPKG_ROOT` set to its directory.
- CMake for C++ consumers.

Rust is required to build the port, not to compile or run applications using the installed library. The port does not install or update Rust automatically.

```powershell
rustup update stable
rustup target add x86_64-pc-windows-msvc
```

## Installation

Clone this repository and run the following from its root:

```powershell
$repo = (Get-Location).Path
$vcpkg = $env:VCPKG_ROOT
& "$vcpkg/vcpkg.exe" install fastlowess:x64-windows "--overlay-ports=$repo/bindings/cpp/vcpkg"
```

The overlay builds the pinned release archive, not the working checkout. Both Debug and Release libraries are installed. Source and Cargo dependency downloads require network access unless already cached.

## CMake Integration

Configure your application with vcpkg's toolchain and `x64-windows` triplet:

```powershell
cmake -S . -B build "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
```

In your application's CMake project:

```cmake
find_package(fastlowess CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE fastlowess::fastlowess)
```

The target supplies the installed include directory, C++17 requirement, and the matching Debug or Release import library. Include `<fastlowess.hpp>` for the C++ interface or `<fastlowess.h>` for the C API. Ensure the matching `fastlowess_cpp.dll` is deployed beside the application or available on `PATH`.

See the [C++ documentation](https://thisisamirv.github.io/lowess-project/cpp/) for API details and smoothing examples.

## Consumer Validation

The included `test-project` is a separate C++ application that uses only the installed package. It checks a linear fit and copies the matching DLL beside its executable.

From the repository root, after installation:

```powershell
$repo = (Get-Location).Path
$vcpkg = $env:VCPKG_ROOT
cmake -S "$repo/bindings/cpp/vcpkg/test-project" -B "$repo/target/vcpkg-consumer" "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=$vcpkg/installed" -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build "$repo/target/vcpkg-consumer" --config Debug
ctest --test-dir "$repo/target/vcpkg-consumer" -C Debug --output-on-failure
cmake --build "$repo/target/vcpkg-consumer" --config Release
ctest --test-dir "$repo/target/vcpkg-consumer" -C Release --output-on-failure
```

If installation used `--x-install-root`, set `VCPKG_INSTALLED_DIR` to that same directory instead of `$vcpkg/installed`.

The port and both consumer configurations passed validation with vcpkg registry revision `13465a7b726f171350defa5368b901a52f7a6af5`. No CRT-check suppression is used.

### clangd

Visual Studio generators do not emit `compile_commands.json`. The test project's local `.clangd` expects a compilation database in `target/vcpkg-consumer-editor`. With LLVM and Ninja on `PATH`, configure it using the same installation prefix:

```powershell
cmake -S "$repo/bindings/cpp/vcpkg/test-project" -B "$repo/target/vcpkg-consumer-editor" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=$vcpkg/installed" -DVCPKG_TARGET_TRIPLET=x64-windows
```

Restart clangd after first configuration if cached diagnostics remain. Build directories and compilation databases are generated locally and are not tracked.

## Port Maintenance

- `fastlowess/vcpkg.json` declares package metadata and supported triplets.
- `fastlowess/portfile.cmake` verifies Rust, downloads the SHA512-pinned source, and installs the package through a thin CMake/Cargo wrapper.
- `fastlowess/Cargo.lock` pins Cargo dependency resolution because the release archive does not include a lockfile. Builds use `--locked`.
- The installed copyright notice describes Rust dependency-license provenance; the lockfile is installed under `share/fastlowess`.

For a version update, refresh the manifest, source checksum, Cargo lockfile, and wrapper project version together, then repeat Debug and Release consumer tests. Do not expand supported triplets without testing them. Offline and vcpkg download-only operation have not been validated.

Rust toolchain provisioning and Cargo downloads remain review considerations for curated-registry inclusion. Registry acceptance follows the [vcpkg maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide).
