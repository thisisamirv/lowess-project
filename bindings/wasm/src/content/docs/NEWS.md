---
title: News
---
<!-- markdownlint-disable MD024 MD025 -->
# fastlowess-wasm (development version)

## Changed

* Updated `typedoc-plugin-markdown` to v4.13.
* `make wasm-dev` now runs `npm update` after `npm install`, so dependencies are kept current.

## Fixed

* Same fix as Node.js: `README.md` is now embedded on the Starlight homepage via `dev/add-readme-to-docs.js`, wired into `npm run docs` and `make wasm-dev`.
* Fixed `concepts.md` figures not rendering: the MkDocs-only `<figure markdown="span">`/attr_list (`{ width="..." }`) syntax isn't supported by Starlight's Markdown renderer, so the image markdown inside was left as raw unprocessed text. Converted all 4 figures to plain `![alt](src)` images with an italicized caption below.
* Fixed inline/display LaTeX math (`$...$`/`$$...$$`) rendering as literal text on the Node.js/WASM docs sites; wired `remark-math`/`rehype-katex` into `astro.config.mjs` and added a KaTeX stylesheet, so the existing math syntax now renders properly.

# fastlowess-wasm 3.1.0

## Added

* Added `npm run lint` to the `Lint` step in `ci-wasm.yml`, so JavaScript source and test files are linted via `oxlint` on every CI run.

## Changed

* Moved WASM documentation from ReadTheDocs to GitHub Pages, served by Starlight at <https://thisisamirv.github.io/lowess-project/wasm/>. The ReadTheDocs site no longer includes WASM-specific content. `dev/add-wasm-outputs.js` runs as part of `make wasm-dev`, executing each JavaScript code block in the docs and injecting its output back into the Markdown source.
* `make wasm` (`default:`) now builds both the Node.js and web WASM targets and links the Node.js package globally via `npm link`. The full dev workflow moves to `make wasm-dev`.
* Updated `oxlint` dependency to 1.80.
* Replace the outdated `jetli/wasm-pack-action` workflow with `taiki-e/install-action`.

# fastlowess-wasm 3.0.0

## Fixed

* Fixed `OnlineLowess.add_point()` returning `undefined` instead of `null` when the sliding window has not yet accumulated enough points.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` getters to `y` and `standard_error`. This is a **breaking change**.
* Updated `oxlint` to v1.79.

# fastlowess-wasm 2.0.0

## Added

* Added `custom_weights` field to `LowessOptions` (passed in the options object to `smooth()`). Accepts a `Float64Array` of non-negative per-observation weights. Batch only.

## Changed

* Renamed all JS-facing option keys to snake_case by removing `#[serde(rename = "camelCase")]` attributes from `SmoothOptions`, `StreamingOptions`, and `OnlineOptions`. JSON passed from JavaScript must now use snake_case keys.
* Updated `Diagnostics` getter names to snake_case: `r_squared`, `effective_df`, `residual_sd`.
* Renamed the `update(x: number, y: number)` method on `OnlineLowess` to `add_point(x: number, y: number)`. This is a **breaking change**.
* Updated `oxlint` dependency to v1.73.

# fastlowess-wasm 1.3.0

## Added

* Upgraded `oxlint` to 1.63.

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed deprecated JavaScript license-audit warnings by replacing the transient `npx license-checker` usage in the `Makefile` with a repo-local Node.js license summary script that still fails on GPL-family licenses.
* Linted the source code.

# fastlowess-wasm 1.2.0

## Added

* Added advanced License Compliance check.
* Added advanced dependency check.
* Added advanced outdated dependency check.
* Added advanced lock file check.
* Added WASM size check.

## Changed

* Switched from `eslint` to `oxlint` to remove vulnerabilities.

## Fixed

* Fixed vulnerabilities.
* Fixed license.

# fastlowess-wasm 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)
* Added an `init_panic_hook` function in `src/lib.rs` to be called by JS users during startup.
* Added JSDoc documentation to `lib.rs`
* Refactored the verbose `Reflect::get` boilerplate using `serde` and `serde-wasm-bindgen`. This allows us to define a Rust struct `SmoothOptions` and have `wasm-bindgen` automatically unpack the JS object into it.

## Changed

* Updated `eslint/js`, `eslint`, `globals`, and `eslint-plugin-html` packages to their latest versions.

# fastlowess-wasm 0.99.9

## Changed

* Package is now available on npm (fastlowess-wasm)

# fastlowess-wasm 0.99.8

## Added

* Initial implementation

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
