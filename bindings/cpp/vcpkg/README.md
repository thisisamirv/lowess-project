# LOWESS for vcpkg

The `fastlowess` overlay port provides Rust-backed C++17 LOWESS smoothing with batch, streaming, and online adapters. It installs the checksum-pinned CPU DLL from the published `v4.1.0` release and exposes the CMake target `fastlowess::fastlowess`. Rust and Cargo are not required to install or consume this port.

This port is available as a repository overlay. It is not yet part of vcpkg's curated registry.

## Supported Configuration

| Setting | Support |
| --- | --- |
| Build host | Windows |
| Target triplets | `x64-windows`, `x64-windows-release` (MSVC ABI) |
| Library linkage | Shared DLL |
| MSVC runtime | Dynamic CRT |
| Backend | CPU |
| Consumer configurations | Debug and Release, both using the Release DLL |

Static libraries, static CRT, ARM, MinGW, UWP, GPU builds, and non-Windows targets are not supported by this port. The manifest excludes unvalidated triplets.

## Requirements

- Visual Studio or Build Tools with the Desktop development with C++ workload and a Windows SDK.
- An existing [vcpkg installation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging), with `VCPKG_ROOT` set to its directory.
- CMake for C++ consumers.

The tagged release did not publish an import library. The port uses the MSVC library manager to generate one from the DLL's verified export list. The DLL itself is not rebuilt or renamed.

## Installation

Clone this repository and run the following from its root:

```powershell
$repo = (Get-Location).Path
$vcpkg = $env:VCPKG_ROOT
& "$vcpkg/vcpkg.exe" install fastlowess:x64-windows "--overlay-ports=$repo/bindings/cpp/vcpkg"
```

The overlay downloads the upstream DLL and matching source archive for headers and license texts. Both downloads are verified by SHA512. Network access is needed unless these artifacts are cached; no crates.io or Rust toolchain downloads occur.

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

The target supplies the installed include directory, C++17 requirement, and import library. All consumer configurations map to the published Release DLL. Include `<fastlowess.hpp>` for the C++ interface. Its matching generated ABI header, `fastlowess.h`, is bundled with the port because the release archive omitted it. Ensure `fastlowess-win32-x64.dll` is deployed beside the application or available on `PATH`.

The native DLL uses the release MSVC runtime. C++ wrapper objects own their own C++ allocations, while Rust-owned buffers are released through the DLL's FFI functions. The port declares `VCPKG_POLICY_ONLY_RELEASE_CRT`; it does not disable CRT-linkage checks. A matching Visual C++ runtime must be available on deployment machines.

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

The prebuilt port passed vcpkg post-build validation for both listed triplets. Debug and Release consumer tests passed against the downloaded DLL using the local vcpkg checkout at registry revision `13465a7b726f171350defa5368b901a52f7a6af5`.

### clangd

Visual Studio generators do not emit `compile_commands.json`. The test project's local `.clangd` expects a compilation database in `target/vcpkg-consumer-editor`. With LLVM and Ninja on `PATH`, configure it using the same installation prefix:

```powershell
cmake -S "$repo/bindings/cpp/vcpkg/test-project" -B "$repo/target/vcpkg-consumer-editor" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=$vcpkg/installed" -DVCPKG_TARGET_TRIPLET=x64-windows
```

Restart clangd after first configuration if cached diagnostics remain. Build directories and compilation databases are generated locally and are not tracked.

## Port Maintenance

- `fastlowess/vcpkg.json` declares package metadata and supported triplets.
- `fastlowess/portfile.cmake` downloads the SHA512-pinned native DLL and matching source archive and installs the package through a thin CMake wrapper.
- `fastlowess/fastlowess.def` records the verified DLL export names used to generate the import library.
- `fastlowess/fastlowess.h` was generated from the exact `v4.1.0` source build with cbindgen. It must be updated alongside the binary and wrapper when their ABI changes.
- The installed copyright notice includes the upstream license texts and Rust dependency-license discovery instructions.

For a version update, refresh the manifest, binary/source checksums, export definition, ABI header, and wrapper project version together, then repeat consumer tests. Do not expand supported triplets without testing them. Offline and vcpkg download-only operation have not been validated.

The `v4.1.0` release does not publish the Cargo lockfile used for its prebuilt binary, so exact Rust dependency-version provenance is unavailable. Future release bundles should include that lockfile, dependency notices, the ABI header, and import libraries. Prebuilt-artifact and licensing acceptance remain subject to the [vcpkg maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide).
