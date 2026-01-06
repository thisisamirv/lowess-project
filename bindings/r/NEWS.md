# rfastlowess 0.99.2

### Changed

- Prepared package for rOpenSci Software Peer Review.
- Renamed main functions to avoid conflicts with base R and align with rOpenSci best practices:
  - `smooth()` -> `fastlowess()`
  - `smooth_online()` -> `fastlowess_online()`
  - `smooth_streaming()` -> `fastlowess_streaming()`
- Updated documentation (README, Vignette, `rfastlowess-package.R`) to reflect the new API and rOpenSci guidelines.
- Updated `codemeta.json` and `DESCRIPTION` metadata.

### Added

- Documentation website using `pkgdown` with automated deployment via GitHub Actions.
- Comprehensive function documentation with examples (`@examples`) and cross-references (`@family`).
- URL validation using `urlchecker` in CI workflow.
- `urlchecker` and `pkgdown` to development dependencies in `Makefile`.
- Rigorous parameter validation for all exported functions to provide clear error messages (Guideline 1.11).
- Expanded test suite (`test-errors.R`) covering edge cases and invalid inputs, achieving >96% coverage.
- Added pkgcheck to CI workflow.

### Fixed

- Documentation URLs (via `urlchecker`).
- Package startup messages (none present, verified).
- Fixed `pkgcheck` workflow to run on host runner (bypassing incompatible Docker image).

### Infrastructure

- Added Codecov CI workflow (`test-coverage.yaml`) and badge.
- Cleaned up project root: consolidated `CHANGELOG.md` to `NEWS.md`, moved `CONTRIBUTING.md` and `_pkgdown.yml` to `.github/`.
- Configured `.Rprofile` defaults for headless environments.
- Added `news()` command instructions to README.

# rfastlowess 0.99.1

### Changed

- Modify package for Bioconductor submission.

# rfastlowess 0.99.0

### Added

- Added support for new features in `fastLowess` v0.4.0, including `NoBoundary` boundary policy and `MAD` and `MAR` scaling methods.

### Changed

- Changed license from AGPL-3.0-or-later to dual MIT OR Apache-2.0.
- Updated the documentation to reflect the latest changes.

# rfastlowess 0.3.0

### Added

- Added the option of installing from R-universe without needing Rust.

### Changed

- Updated to `fastLowess` v0.3.0 and `lowess` v0.6.0.
- Updated cross-validation API to use tuple constructors (`KFold`, `LOOCV`).

### Fixed

- Automated vendor checksum fixing for CI builds (CRLF→LF line ending issues).

# rfastlowess 0.2.0

- For changes to the core logic and the API, see the [lowess](https://github.com/thisisamirv/lowess) and [fastLowess](https://github.com/thisisamirv/fastLowess) crates.

### Added

- Added support for new features in `fastLowess` v0.2.0.

### Changed

- Improved documentation.

# rfastlowess 0.1.0

### Added

- Added the R binding for `fastLowess`.
