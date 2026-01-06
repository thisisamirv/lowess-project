# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0]

### Added

- Added support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD` and `MAR` scaling methods.

### Changed

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
- Updated the documentation to reflect the latest changes.

## [0.3.1]

### Added

- Improved documentation.

## [0.3.0]

### Added

- Added `return_residuals` argument to streaming adapter.
- Added `zero_weight_fallback` argument to streaming and online adapters.

### Changed

- Updated `fastLowess` dependency to v0.3.0, incorporating core engine improvements.
- Refactored internal API usage to align with new `fastLowess` crate module structure.
- Updated `smooth()` cross-validation parameter handling to use `KFold` and `LOOCV` tuple constructors.
- Comprehensive documentation updates in `README.md` highlighting robustness features and new benchmarks.
- Updated example scripts (`batch_smoothing.py`, `online_smoothing.py`, `streaming_smoothing.py`) to save plots to `examples/plots/` and improved parameter tuning.

### Fixed

- Fixed documentation build errors.
- Fixed a bug where the `parallel` argument was not exposed in the python binding.

## [0.2.0]

- For changes to the core logic and the API, see the [lowess](https://github.com/av746/lowess) and [fastLowess](https://github.com/av746/fastLowess) crates.

### Added

- Added support for new features in `fastLowess` v0.2.0.

### Changed

- Updated the documentation to reflect the latest changes.
- Changed the module name from `fastLowess` to `fastlowess`.

## [0.1.1]

### Added

- Added support for Python 3.14.

### Changed

- Updated the documentation to reflect the latest changes.

## [0.1.0]

### Added

- Added the python binding for `fastLowess`.

## [0.1.0]

### Added

- Added the python binding for `fastLowess`.
