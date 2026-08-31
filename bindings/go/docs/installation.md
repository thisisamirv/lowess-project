# Installation

## Requirements

- Go 1.21+
- `CGO_ENABLED=1` and a C compiler:
  - Linux/macOS: GCC or Clang (usually already present)
  - Windows: a MinGW-w64 toolchain (e.g. via [MSYS2](https://www.msys2.org/) or [WinLibs](https://winlibs.com/)), since Go's `cgo` invokes `gcc` on Windows, not MSVC's `cl.exe`
- A prebuilt copy of the native `fastlowess_go` static library and its header (`fastlowess_go.h`)

## Within the `lowess-project` monorepo

If you're working inside a checkout of [`lowess-project`](https://github.com/thisisamirv/lowess-project), the root `Makefile` handles building the native library for you:

```sh
make go        # build the Rust FFI crate, then `go build ./...`
make go-dev    # full dev checks: fmt, lint, tests, doc snippets
```

`bindings/go/fastlowess/ffi.go`'s `#cgo` directives point at `../../../target/release-c` (Linux/macOS) or `../../../target/x86_64-pc-windows-gnu/release-c` (Windows), which is where `cargo build -p fastlowess-go --profile release-c` places the static library within this repo's layout.

## As a standalone module

Outside the monorepo, download the prebuilt static library and header attached to a [GitHub release](https://github.com/thisisamirv/lowess-project/releases), then point `cgo` at them:

```sh
export CGO_CFLAGS="-I/path/to/fastlowess_go/include"
export CGO_LDFLAGS="-L/path/to/fastlowess_go/lib -lfastlowess_go -lm -ldl -lpthread"  # Linux
go build ./...
```

On macOS, drop `-ldl -lpthread` (not needed). On Windows, use `-lws2_32 -luserenv -lbcrypt -lntdll -lpthread` instead, and ensure a MinGW-w64 `gcc.exe` is on `PATH`.

Alternatively, build the native library yourself from the [`lowess-project`](https://github.com/thisisamirv/lowess-project) source:

```sh
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project
cargo build -p fastlowess-go --profile release-c
```

## GPU backend

GPU acceleration is available via the native library's `gpu` Cargo feature (`wgpu`: Vulkan/Metal/DX12), but is not enabled in the default build. See the C++ binding's [GPU Backend guide](https://thisisamirv.github.io/lowess-project/cpp/gpu-backend.html) for the underlying feature; building `fastlowess-go` with `--features gpu` produces a Go-compatible library with `GPUEnabled()` returning `true`.
