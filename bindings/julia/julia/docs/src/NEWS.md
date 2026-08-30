<!-- markdownlint-disable MD024 MD025 -->
# FastLOWESS.jl (development version)

## Fixed

* Fixed the Documenter site homepage being a separately maintained `docs/src/index.md` instead of the top-level `README.md`. `make.jl` now regenerates `index.md` from `README.md` before every build, and the stale static copy was removed.
* Fixed the README's raw `<p align="center">` badge/logo HTML blocks rendering as literal text on the Documenter site (unlike GitHub/pkgdown/Starlight/Doxygen); `make.jl` now converts them to plain Markdown image/link syntax before writing `index.md`.
* Fixed the README's `<!-- markdownlint-disable ... -->` comment rendering as literal text on the Documenter site; `make.jl` now strips HTML comments before writing `index.md`.

# FastLOWESS.jl 3.1.0

## Added

* `release-julia-register.yml` now automatically extracts the matching changelog section and appends it as release notes in the JuliaRegistrator comment, enabling auto-merge on major version bumps.

## Changed

* Moved Julia documentation from ReadTheDocs to GitHub Pages, served by Documenter.jl at <https://thisisamirv.github.io/lowess-project/julia/stable/>. The ReadTheDocs site no longer includes Julia-specific content. Code blocks use Documenter.jl `@example` sections, which execute and embed output automatically during the docs build.
* `make julia` (`default:`) now builds the Rust library and installs the Julia package via `Pkg.develop`. The full dev workflow moves to `make julia-dev`.

# FastLOWESS.jl 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published JLL artifacts. Run `install_gpu()` to download a prebuilt GPU library, or build locally with `cargo build --release --features gpu`.

## Fixed

* Fixed `LowessResult.iterations_used` returning the raw FFI sentinel `-1` instead of `nothing` when robustness iterations were not applicable.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Removed `dev/format_julia.jl`; formatting is now inlined in `bindings/julia/Makefile`.

# FastLOWESS.jl 2.0.0

## Added

* Added `custom_weights` keyword argument to `fit(model, x, y; custom_weights)`. Accepts a `Vector{Float64}` of non-negative per-observation weights. Batch only.

## Changed

* Replaced `add_points(online, x::Vector{Float64}, y::Vector{Float64}) :: LowessResult` with `add_point(online, x::Float64, y::Float64) :: Union{Float64, Nothing}`. The function now processes a single point and returns the smoothed value, or `nothing` if not enough points have been accumulated yet. The underlying C FFI symbol is renamed from `jl_online_lowess_add_points` to `jl_online_lowess_add_point`. This is a **breaking change**.

# FastLOWESS.jl 1.3.0

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed Windows local Julia runs by exporting an absolute `FASTLOWESS_LIB` path from the `Makefile` and moving DLL discovery in `FastLOWESS.jl` to runtime (`__init__()` plus runtime `ccall`), preventing stale precompiled library paths from being reused.
* Linted the source code.

# FastLOWESS.jl 1.2.0

## Fixed

* Linted examples.

# FastLOWESS.jl 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)

## Changed

* Wrapped all FFI functions in std::panic::catch_unwind. This ensures that if the Rust library panics (e.g., due to an internal assertion), it will be caught and reported as an error to Julia.

# FastLOWESS.jl 0.99.9

## Changed

* Package is now registered on JuliaRegistries

# FastLOWESS.jl 0.99.8

## Added

* Initial implementation

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
