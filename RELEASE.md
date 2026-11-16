fastLowess 5.1.0 clarifies Online point lifecycle across the Rust core and every language binding, hardens Streaming and Online input handling, and renames the C++ C ABI to a dedicated `fastlowess_*` prefix.

## Highlights

- **Explicit Online `add_point` status:** Adding a point now returns whether the window is still filling, a smoothed estimate is ready, or the observation was dropped under `missing = "drop"`. Bindings no longer infer that distinction from `None` / `null` / empty results.
- **Safer Streaming and Online inputs:** Online fits sort the sliding window by `x` before each update so non-monotonic streams stay correct. Streaming rejects overlap merges that are not non-decreasing in `x`, and several bindings reject invalid lengths, negative iteration counts, and mismatched custom weights earlier.
- **Clearer C++ / Go native ownership:** Predict handles can be taken explicitly before result free, remaining handles are freed safely, and Online status is exposed through dedicated enums / status codes instead of overloaded `has_value` sentinels alone.
- **Idempotent close / dispose paths:** Node.js, Python, R, and WASM add `close()` (and where applicable context-manager / `Symbol.dispose` support) on Online, Streaming, and retained-model result types so native state can be released deliberately.

## Breaking Changes

- **Online `add_point` return type** changes in Rust and all bindings:
  - Rust: `AddPointResult` (`Filling` / `Ready` / `Dropped`) instead of `Option<OnlineOutput>`
  - C++ / Go / Java / Julia / R / Python / Node.js / WASM: status + optional point instead of a bare optional / nullable online output
- **C++ C ABI rename:** public symbols move from the generic `cpp_*` prefix to `fastlowess_*` (for example `fastlowess_lowess_new`). Rebuild consumers against the updated header and library.
- Binding-specific API details differ; check the binding NEWS / changelog section before upgrading.

## Other Notable Fixes

- Streaming no longer panics when derivatives were requested but missing; it returns an error instead.
- Extend/Zero boundary pads use unit spacing when leading/trailing `x` ties would otherwise stack pads.
- C++ honors `LowessOptions::custom_weights` when fit-argument weights are empty, and bounds-checks indexed result accessors.
- Go keeps slice inputs alive across cgo, makes `Close`/finalizer races safer, and takes predict-handle ownership explicitly.
- Java closes retained `PredictModel` handles through take-once ownership with a `Cleaner` safety net.
- Julia uses `usize` / `Csize_t` for FFI lengths on Windows LLP64 and frees native results more safely.
- Node.js / Python / WASM preserve clearer error categories and align async/custom-weight / Online failure mapping with Batch/`predict`.
