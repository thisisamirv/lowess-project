# rfastlowess

## Test environments

- Local Linux (Ubuntu/Debian)
- GitHub Actions: Ubuntu-latest, macOS-latest, macOS-14 (ARM64), Windows-latest

## R CMD check results

0 errors | 0 warnings | 0 notes

## Submissions

This is a new submission.

## Comments

- This package provides Rust-powered LOWESS smoothing with parallel execution.
- Rust dependencies are vendored in `src/vendor.tar.xz` for self-contained builds.
- The `configure` script handles platform-specific linker flags for Linux and macOS.
