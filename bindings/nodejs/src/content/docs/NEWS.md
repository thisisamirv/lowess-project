---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (Node.js) (development version)

## Changed

* Updated `typedoc-plugin-markdown` to v4.13.
* `make nodejs-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

## Fixed

* Fixed the docs homepage's "Get Started" button jumping straight to the Installation page without ever showing the README content. `README.md` is now embedded on the Starlight homepage below the hero via a new `dev/add-readme-to-docs.js` script, wired into both `npm run docs` and `make nodejs-dev`.

# fastlowess (Node.js) 3.1.0

## Added

* Added `npm run lint` to the `Lint` step in `ci-nodejs.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

## Changed

* Moved Node.js documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/lowess-project/nodejs/>. The ReadTheDocs site no longer includes Node.js-specific content. `dev/add-nodejs-outputs.js` runs as part of `make nodejs-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
* `make nodejs` (`default:`) now builds the native addon and links it globally via `npm link`. The full dev workflow moves to `make nodejs-dev`.
* Updated `oxlint` dependency to 1.80.

# fastlowess (Node.js) 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published npm binaries. Run `await fastlowess.installGpu()` to download a prebuilt GPU addon (requires restarting Node.js), or build locally with `napi build --features gpu`.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` fields to `y` and `standard_error`. This is a **breaking change**.
* Updated `@napi-rs/cli` to v3.8 and `oxlint` to v1.79.

# fastlowess (Node.js) 2.0.0

## Added

* Added `OnlineOutput` object to the Node.js binding. `OnlineLowess.add_point()` now returns `OnlineOutput | null` instead of `number | null`, exposing `smoothed`, `std_error`, `residual`, `robustness_weight`, and `iterations_used`.
* Added `return_se` and `cv_seed` fields to `SmoothOptions`.
* Added `customWeights` as an optional per-call argument to `fit(x, y, customWeights?)` and `fit_async(x, y, customWeights?)`. Accepts a `Float64Array` of non-negative per-observation weights. Includes pre-flight length-mismatch and non-negative validation. Batch only.
* Added JavaScript-layer option key validation: unknown keys in `SmoothOptions`, `StreamingOptions`, or `OnlineOptions` now throw a `TypeError` listing all valid keys, via wrapper classes around the native NAPI exports.

## Changed

* Renamed all `Diagnostics`, `SmoothOptions`, `StreamingOptions`, and `OnlineOptions` interface fields to snake_case (`r_squared`, `effective_df`, `residual_sd`, `chunk_size`, `merge_strategy`, `window_capacity`, `min_points`, `update_mode`, and all smoothing option fields).
* Renamed binding methods to snake_case: `fit_async`, `process_chunk`, `add_points`.
* Renamed `LowessResultObj` getters to snake_case: `standard_errors`, `confidence_lower`, `confidence_upper`, `prediction_lower`, `prediction_upper`, `robustness_weights`, `cv_scores`, `fraction_used`, `iterations_used`.
* Updated `index.d.ts` to reflect all renamed fields and methods.
* Replaced `add_points(x: Float64Array, y: Float64Array): LowessResultObj` on `OnlineLowess` with `add_point(x: number, y: number): OnlineOutput | null`. The method now processes a single point and returns an `OnlineOutput` object, or `null` if not enough points have been accumulated yet. This is a **breaking change**.
* Changed default `OnlineOptions.window_capacity` from 100 to 1000 and `OnlineOptions.min_points` from 2 to 3, matching the defaults used by the loess binding.
* `OnlineLowess` now forwards all `SmoothOptions` fields to the underlying builder (previously only `fraction`, `iterations`, and `parallel` were forwarded; all other fields were hardcoded to `None`/`false`).
* Updated `napi-rs/cli` dependency to v3.7 and `oxlint` to v1.73.

# fastlowess (Node.js) 1.3.0

## Added

* Upgraded `oxlint` to 1.63.
* Upgraded `napi-rs/cli` to 3.6.

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed `make nodejs` on Windows when `/bin/bash` could not launch `npm` from `C:/Program Files/nodejs` by using `npm.cmd`/`npx.cmd` in the `Makefile`.
* Fixed deprecated JavaScript license-audit warnings by replacing the transient `npx license-checker` usage in the `Makefile` with a repo-local Node.js license summary script that still fails on GPL-family licenses.
* Fixed `.build()` errors incorrectly using `Status::GenericFailure`; they now return `Status::InvalidArg` since build failures originate from invalid configuration (accumulated parse errors), not from runtime execution.
* Linted the source code.

# fastlowess (Node.js) 1.2.0

## Added

* Added advanced License Compliance check.
* Added advanced dependency check.
* Added advanced outdated dependency check.
* Added advanced lock file check.
* Added advanced TypeScript check.

## Changed

* Switched from `eslint` to `oxlint` to remove vulnerabilities.

## Fixed

* Fixed vulnerabilities.

# fastlowess (Node.js) 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)
* Added JSDoc documentation to `lib.rs` for napi-rs generation
* Added asynchronous support for batch processing

## Changed

* Updated `eslint/js`, `eslint`, and `globals` packages to their latest versions.

# fastlowess (Node.js) 0.99.9

## Changed

* Package is now available on npm (fastlowess)

# fastlowess (Node.js) 0.99.8

## Added

* Initial implementation

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
