# LOWESS for vcpkg

The `fastlowess` overlay port provides Rust-backed C++17 LOWESS smoothing with batch, streaming, and online adapters. It installs a SHA512-verified prebuilt CPU library from the matching upstream release and exposes the `unofficial-fastlowess` CMake package with the `unofficial::fastlowess::fastlowess` target. Rust and Cargo are not required to install or consume this port.

Use this port as a repository overlay. Its supported targets are limited to those accepted by the portfile and manifest and for which the matching upstream release publishes a binary.

## Supported Configuration

| Setting | Support |
| --- | --- |
| Target triplets | Windows x64/ARM64, Linux x64/ARM64, and macOS x64/ARM64 |
| Linux libc | glibc only |
| Library linkage | Shared library only |
| Windows runtime | Dynamic CRT |
| Backend | CPU |
| Consumer configurations | Debug and Release, both using the upstream prebuilt library |

Static linkage, static CRT, musl, 32-bit and ARMv7 targets, MinGW, UWP, Android, iOS, GPU builds, and other operating systems are not supported by this curated port. Linux and macOS require dynamic-linkage triplets. The Linux target compiler is checked for glibc during configuration, so custom triplet names cannot silently select an incompatible binary. Use an overlay port for musl. The manifest and available release binaries are authoritative; do not assume a triplet is supported just because its architecture is listed above.

## Requirements

- A C++ toolchain for the selected target: Visual Studio/Windows SDK, Linux C++ toolchain, or Xcode command-line tools.
- An existing [vcpkg installation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started-packaging), with `VCPKG_ROOT` set to its directory.
- CMake for C++ consumers.

The port uses the MSVC library manager to generate an import library from the verified export list so it references the published DLL filename. The upstream native binaries are not rebuilt or renamed.

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

The overlay downloads a SHA512-verified platform archive containing the upstream native library, C++ headers, generated ABI header, and third-party dependency license report. It also downloads the matching source archive for the MIT and Apache license texts; that archive is SHA512-verified as well. Network access is needed unless these artifacts are cached; no crates.io or Rust toolchain downloads occur.

## CMake Integration

For example, configure a Windows application with vcpkg's toolchain and the triplet used for installation:

```powershell
cmake -S . -B build "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows
```

On Linux or macOS, use the same form with a matching dynamic triplet. The curated Linux port supports glibc only; use an overlay port to select a musl release binary.

In your application's CMake project:

```cmake
find_package(unofficial-fastlowess CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE unofficial::fastlowess::fastlowess)
```

The target supplies the installed include directory, C++17 requirement, and native library. All consumer configurations map to the selected prebuilt artifact. Include `<fastlowess.hpp>` for the C++ interface; the port installs the matching generated ABI header alongside it. On Windows, make the installed DLL available to the application through its executable directory or `PATH`. On Unix-like systems, ensure the platform loader can find the installed shared library.

On Windows, the native DLL uses the MSVC runtime. C++ wrapper objects own their C++ allocations, while Rust-owned buffers are released through the DLL's FFI functions. The port declares `VCPKG_POLICY_ONLY_RELEASE_CRT`; it does not disable CRT-linkage checks. A matching Visual C++ runtime must be available on deployment machines.

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

For Linux or macOS, use the matching dynamic triplet. For example, on glibc Linux:

```sh
cmake -S bindings/cpp/vcpkg/test-project -B target/vcpkg-consumer-linux \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET=x64-linux-dynamic
cmake --build target/vcpkg-consumer-linux
ctest --test-dir target/vcpkg-consumer-linux --output-on-failure
```

To validate an update, install the port for each supported triplet, then build and run the consumer in Debug and Release where the toolchain supports both configurations. Include at least one linear-fit case and verify the duplicate-include behavior. Record which operating systems, architectures, libc variants, compilers, and configurations were actually tested; do not infer test coverage from artifact availability.

### clangd

Visual Studio generators do not emit `compile_commands.json`. The test project's local `.clangd` expects a compilation database in `target/vcpkg-consumer-editor`. With LLVM and Ninja on `PATH`, configure it using the same installation prefix:

```powershell
cmake -S "$repo/bindings/cpp/vcpkg/test-project" -B "$repo/target/vcpkg-consumer-editor" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=clang++ "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=$vcpkg/installed" -DVCPKG_TARGET_TRIPLET=x64-windows
```

Restart clangd after first configuration if cached diagnostics remain. Build directories and compilation databases are generated locally and are not tracked.

## Port Maintenance

- `fastlowess/vcpkg.json` declares package metadata and supported triplets.
- `fastlowess/portfile.cmake` downloads the SHA512-pinned platform archive and matching source archive and installs the package through a thin CMake wrapper.
- `fastlowess/fastlowess.def` records the verified DLL export names used to generate the import library.
- Upstream release archives include the generated ABI header and `THIRD_PARTY_LICENSES.html`, generated from the committed workspace lockfile using `dev/about.toml` and `dev/about.hbs`. The report documents third-party Rust dependency licenses and provenance. The port installs it alongside the upstream MIT and Apache license texts.

The v5.0.0 SHA512 sentinels in `portfile.cmake` must be replaced with the published archive and source-archive hashes before installation or submission.

For each release update, refresh the manifest, archive/source checksums, supported triplet logic, export definition, and wrapper project version together. Confirm the platform archive contains the matching ABI header and dependency license report, then repeat consumer tests for every supported triplet. Do not expand supported triplets without testing them. Offline and vcpkg download-only operation have not been validated. Follow the [vcpkg maintainer guide](https://learn.microsoft.com/en-us/vcpkg/contributing/maintainer-guide) for package acceptance requirements.
