<!-- markdownlint-disable MD024 MD025 -->
# fastlowess (Python) 3.1.0

## Changed

* Migrated Python documentation from MkDocs to Sphinx (with MyST-Parser and jupyter-sphinx). Code blocks now execute and embed output automatically via `jupyter-sphinx`.
* `make python` (`default:`) now installs to the user Python environment via `pip install --user`. The full dev workflow (venv setup, formatting, linting, testing, doc-snippet verification) moves to `make python-dev`.

## Fixed

* Enforced keyword-only arguments beyond the first positional allowance in `Lowess`, `StreamingLowess`, and `OnlineLowess`, matching R's behaviour: `Lowess(fraction, *, ...)`, `StreamingLowess(fraction, chunk_size, *, ...)`, `OnlineLowess(fraction, window_capacity, min_points, *, ...)`. The `.pyi` stubs were updated with the same `*` separator.

# fastlowess (Python) 3.0.0

## Added

* Exposed a `backend` option (`"cpu"` default, `"gpu"`) on `Lowess`, gated behind an opt-in `gpu` Cargo feature not enabled in published wheels. Run `fastlowess.install_gpu()` to download a prebuilt GPU wheel, or build locally with `maturin develop --features gpu`.

## Changed

* Renamed `OnlineOutput`'s `smoothed` and `std_error` properties to `y` and `standard_error`. This is a **breaking change**.

# fastlowess (Python) 2.0.0

## Added

* Added `OnlineOutput` class to the Python binding. `OnlineLowess.add_point()` now returns `OnlineOutput | None` instead of `float | None`, exposing `smoothed`, `std_error`, `residual`, `robustness_weight`, and `iterations_used`.
* Added `custom_weights` parameter to the `Lowess.fit(x, y, custom_weights=None)` method. Accepts a `list[float]` of non-negative per-observation weights. Batch only.
* Removed `smooth()`, `smooth_streaming()`, and `smooth_online()` convenience function stubs from `_core.pyi`.

## Changed

* Renamed the `update(x, y)` method on `OnlineLowess` to `add_point(x, y)` and removed the separate array-based `add_points(x, y)` method. `add_point` processes a single point and returns the smoothed value as `float | None`. This is a **breaking change**.
* Updated `pyo3` and `numpy` dependencies to v0.29.

# fastlowess (Python) 1.3.0

## Changed

* Updated MSRV to 1.89 to access the significant improvements made in `wide` since version 0.7.

## Fixed

* Fixed `make python` failing when `ruff` is not installed globally by bootstrapping `ruff` inside the Python virtual environment before formatting and linting.
* Fixed `make python` on Windows by selecting the correct virtual environment activation script (`.venv/Scripts/activate` instead of the Unix-only `.venv/bin/activate`).
* Fixed the Python public API to actually accept documented array-like inputs by coercing `Lowess.fit()`, `StreamingLowess.process_chunk()`, and `OnlineLowess.add_points()` arguments via `np.asarray(..., dtype=np.float64)` before calling the native extension.
* Fixed Python wrapper analyzer issues by switching native extension lookups to runtime imports, avoiding wrapper class name shadowing in `TYPE_CHECKING`, and adding explicit wrapper docstrings.
* Fixed false-positive Pylint warnings in `bindings/python/python/fastlowess/_core.pyi` by marking stub-only ellipsis bodies and signature arguments as intentional.

# fastlowess (Python) 1.2.0

## Changed

* Updated `pyo3` to v0.28 from v0.27.
* Updated `numpy` to v0.28 from v0.27.

# fastlowess (Python) 1.0.0

## Added

* Added `mean` scaling method (Mean Absolute Deviation)

## Changed

* Wrapped the heavy computation logic in `py.allow_threads` to allow Python to release the GIL during computation.

# fastlowess (Python) 0.99.7

## Changed

* Switch to Stable ABI for CPython

# fastlowess (Python) 0.4.0

## Added

* Support for new features in `fastLowess` v0.4.0

## Changed

* Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0
* Updated documentation

# fastlowess (Python) 0.3.0

## Changed

* Updated `fastLowess` dependency to v0.3.0
* Refactored internal API usage
* Updated cross-validation parameter handling

## Fixed

* Documentation build errors
* Bug where `parallel` argument was not exposed

# fastlowess (Python) 0.2.0

## Added

* Support for new features in `fastLowess` v0.2.0

## Changed

* Updated documentation
* Changed module name from `fastLowess` to `fastlowess`

# fastlowess (Python) 0.1.0

## Added

* Python binding for `fastLowess`
* Support for Python 3.14

For the full changelog, see:
<https://github.com/thisisamirv/lowess-project/blob/main/CHANGELOG.md>
