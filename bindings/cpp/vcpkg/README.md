# LOWESS for vcpkg

The `fastlowess` overlay port provides Rust-backed C++17 LOWESS smoothing with batch, streaming, and online adapters. It installs a checksum-pinned CPU library from the published `v4.1.0` release and exposes the CMake target `fastlowess::fastlowess`. Rust and Cargo are not required to install or consume this port.

This port is available as a repository overlay. It is not yet part of vcpkg's curated registry.

## Supported Configuration

| Setting | Support |
| --- | --- |
| Target triplets | Windows: `x64-windows`, `x64-windows-release`, `arm64-windows`; Linux: `x64-linux-dynamic`, `arm64-linux-dynamic`; macOS: `x64-osx-dynamic`, `arm64-osx-dynamic` |
| Linux libc | Triplet names containing `musl` select musl artifacts; other Linux triplets select glibc artifacts |
| Library linkage | Shared library only |
| Windows runtime | Dynamic CRT |
| Backend | CPU |
| Consumer configurations | Debug and Release, both using the upstream Release library |

Static linkage, static CRT, 32-bit and ARMv7 targets, MinGW, UWP, Android, iOS, GPU builds, and other operating systems are not supported. The standard Linux and macOS triplets that default to static linkage are excluded; use dynamic-linkage triplets. The manifest also excludes unvalidated architectures.

## Requirements

- A C++ toolchain for the selected target: Visual Studio/Windows SDK, Linux C++ toolchain, or Xcode command-line tools.
- An existing [vcpkg installation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging), with `VCPKG_ROOT` set to its directory.
- CMake for C++ consumers.

The tagged release did not publish Windows import libraries. The port uses the MSVC library manager to generate them from the DLL's verified export list. Native binaries are not rebuilt or renamed.

## Installation

Clone this repository and run the following from its root:

```powershell
$repo = (Get-Location).Path
$vcpkg = $env:VCPKG_ROOT
& "$vcpkg/vcpkg.exe" install fastlowess:x64-windows "--overlay-ports=$repo/bindings/cpp/vcpkg"
```

For Linux and macOS, use a matching dynamic-linkage triplet, for example:

```sh
vcpkg install fastlowess:x64-linux-dynamic --overlay-ports=bindings/cpp/vcpkg
vcpkg install fastlowess:arm64-osx-dynamic --overlay-ports=bindings/cpp/vcpkg
```

The overlay downloads the upstream native library and matching source archive for headers and license texts. Both downloads are verified by SHA512. Network access is needed unless these artifacts are cached; no crates.io or Rust toolchain downloads occur.

## CMake Integration

For example, configure a Windows application with vcpkg's toolchain and the triplet used for installation:

```powershell
cmake -S . -B build "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
```

On Linux or macOS, use the same form with the matching dynamic triplet, such as `x64-linux-dynamic` or `arm64-osx-dynamic`.

In your application's CMake project:

```cmake
find_package(fastlowess CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE fastlowess::fastlowess)
```

The target supplies the installed include directory, C++17 requirement, and native library. All consumer configurations map to the published Release artifact. Include `<fastlowess.hpp>` for the C++ interface. Its matching generated ABI header, `fastlowess.h`, is bundled with the port because the release archive omitted it. On Windows, deploy the matching `fastlowess-win32-*.dll` beside the application or make it available on `PATH`; Unix consumers use the installed `.so` or `.dylib`.

On Windows, the native DLL uses the release MSVC runtime. C++ wrapper objects own their C++ allocations, while Rust-owned buffers are released through the DLL's FFI functions. The port declares `VCPKG_POLICY_ONLY_RELEASE_CRT`; it does not disable CRT-linkage checks. A matching Visual C++ runtime must be available on deployment machines.

See the [C++ documentation](https://thisisamirv.github.io/lowess-project/cpp/) for API details and smoothing examples.

## Consumer Validation

The included `test-project` is a separate C++ application that uses only the installed package. It checks a linear fit and copies the matching native library beside its executable.

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

For Linux or macOS, use the matching dynamic triplet. For example, on Linux:

```sh
cmake -S bindings/cpp/vcpkg/test-project -B target/vcpkg-consumer-linux \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux-dynamic
cmake --build target/vcpkg-consumer-linux
ctest --test-dir target/vcpkg-consumer-linux --output-on-failure
```

The prebuilt port passed vcpkg post-build validation and Debug/Release consumer tests on Windows x64, and passed vcpkg post-build validation plus the Linux x64 dynamic-triplet consumer smoke test. Other listed architecture artifacts are built by the upstream release workflow but have not been tested locally in this environment.

### clangd

Visual Studio generators do not emit `compile_commands.json`. The test project's local `.clangd` expects a compilation database in `target/vcpkg-consumer-editor`. With LLVM and Ninja on `PATH`, configure it using the same installation prefix:

```powershell
cmake -S "$repo/bindings/cpp/vcpkg/test-project" -B "$repo/target/vcpkg-consumer-editor" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=$vcpkg/installed" -DVCPKG_TARGET_TRIPLET=x64-windows
```

Restart clangd after first configuration if cached diagnostics remain. Build directories and compilation databases are generated locally and are not tracked.

## Port Maintenance

- `fastlowess/vcpkg.json` declares package metadata and supported triplets.
- `fastlowess/portfile.cmake` downloads the SHA512-pinned platform library and matching source archive and installs the package through a thin CMake wrapper.
- `fastlowess/fastlowess.def` records the verified DLL export names used to generate the import library.
- `fastlowess/fastlowess.h` was generated from the exact `v4.1.0` source build with cbindgen. It must be updated alongside the binary and wrapper when their ABI changes.
- The installed copyright notice includes the upstream license texts and Rust dependency-license discovery instructions.

For a version update, refresh the manifest, binary/source checksums, export definition, ABI header, and wrapper project version together, then repeat consumer tests. Do not expand supported triplets without testing them. Offline and vcpkg download-only operation have not been validated.

The `v4.1.0` release does not publish the Cargo lockfile used for its prebuilt binaries, so exact Rust dependency-version provenance is unavailable. Future release bundles should include that lockfile, dependency notices, and matching headers/import libraries. Prebuilt-artifact and licensing acceptance remain subject to the [vcpkg maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide).
