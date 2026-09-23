# LOWESS reference implementations

These files are standalone historical/reference implementations used to check
`lowess` and `fastLowess`. They are not linked into the production crates or
bindings.

## `r-lowess.c`

A self-contained extraction of the numerical engine used by R's
`stats::lowess()`:

- `lowest()` local weighted regression and `clowess()` iteration logic from
  R's `src/library/stats/src/lowess.c`;
- local equivalents of `fmax2()`, `imin2()`, and `imax2()` from R Mathlib;
- the double-precision `rPsort()` specialization and comparison semantics from
  R's `src/main/sort.c`;
- a standalone allocation/validation entry point, `r_lowess()`, so the file
  compiles without R headers or the R runtime.

Inputs must already be sorted by ascending `x`. The implementation uses
`double`, R's delta interpolation order, normalized adjusted local weights,
and R's effective-zero robustness condition
`cmad < 1e-7 * mean(abs(residuals))`.

R is distributed under GPL-2-or-later, so this derived reference file is
provided under the same terms.

Compile-check it with:

```sh
gcc -std=c11 -Wall -Wextra -Werror -pedantic -c r-lowess.c
```

## `cleveland-lowess.f`

W. S. Cleveland's December 1985 Netlib implementation in fixed-form Fortran.
It contains the original `lowess`, `lowest`, and shell-sort routines in one
file. It uses default `REAL` arithmetic (normally single precision) and expects
the caller to provide robustness and residual work arrays.

This historical implementation differs from R's later C implementation in
several details, including neighborhood traversal, local regression arithmetic,
median sorting, and the absence of R's effective-zero robustness stop. Keep it
as an algorithm/provenance reference rather than an exact oracle for
`stats::lowess()`.

Compile-check it with:

```sh
gfortran -std=legacy -Wall -Wextra -Wno-compare-reals \
  -c cleveland-lowess.f
```

The exact `x(j) == x(i)` comparison is intentional and comes from the
historical algorithm.
