---
title: "Installation"
weight: 15
---

## Requirements

- Java 17+ (this binding targets Java 25 LTS by default; see the [pom.xml](https://github.com/thisisamirv/lowess-project/blob/main/bindings/java/pom.xml) `maven.compiler.release` property)
- A prebuilt copy of the native `fastlowess_java` library (JNI shared library) on the JVM's native library search path

## Within the `lowess-project` monorepo

If you're working inside a checkout of [`lowess-project`](https://github.com/thisisamirv/lowess-project), the root `Makefile` handles building the native library for you:

```sh
make java        # build the Rust JNI crate, then `mvn package`
make java-dev     # full dev checks: fmt, clippy, export verification, mvn test
```

`NativeBridge` loads the native library from the path given by the `fastlowess.native.dir` system property (or `-Dfastlowess.native.dir=...`), which defaults to `../../target/debug` relative to `bindings/java` — where `cargo build -p fastlowess-java` places the shared library within this repo's layout.

## As a standalone dependency

Outside the monorepo, add the Maven coordinate to your `pom.xml`:

```xml
<dependency>
    <groupId>com.thisisamirv</groupId>
    <artifactId>fastlowess</artifactId>
    <version>3.1.0</version>
</dependency>
```

Then point the JVM at a prebuilt native library, either via the `fastlowess.native.dir` system property or by placing it on `java.library.path`:

```sh
java -Dfastlowess.native.dir=/path/to/native/lib -cp your-app.jar com.example.Main
```

Prebuilt native libraries are attached to [GitHub releases](https://github.com/thisisamirv/lowess-project/releases) for common platforms (`fastlowess_java.dll` on Windows, `libfastlowess_java.so` on Linux, `libfastlowess_java.dylib` on macOS).

Alternatively, build the native library yourself from the [`lowess-project`](https://github.com/thisisamirv/lowess-project) source:

```sh
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project
cargo build -p fastlowess-java --release
```

## GPU backend

GPU acceleration is available via the native library's `gpu` Cargo feature (`wgpu`: Vulkan/Metal/DX12), but is not enabled in the default build. See the C++ binding's [GPU Backend guide](https://thisisamirv.github.io/lowess-project/cpp/gpu-backend.html) for the underlying feature; building `fastlowess-java` with `--features gpu` produces a library where `FastLowess.gpuEnabled()` returns `true`.
