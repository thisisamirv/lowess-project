/*
 * Self-contained reference implementation of R's stats::lowess engine.
 *
 * Numerical code is derived from R's src/library/stats/src/lowess.c.
 * The local helpers below replace dependencies on Rmath, R_ext/Boolean.h,
 * and R_ext/Utils.h so this file can be compiled without the R runtime.
 *
 * R is free software distributed under GPL-2-or-later. This reference file
 * is therefore provided under the same terms. See https://www.r-project.org/.
 */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef enum { R_LOWESS_FALSE = 0, R_LOWESS_TRUE = 1 } r_lowess_boolean;

static double r_lowess_fmax2(double x, double y) {
  if (isnan(x) || isnan(y))
    return x + y;
  return (x < y) ? y : x;
}

static int r_lowess_imin2(int x, int y) { return (x < y) ? x : y; }

static int r_lowess_imax2(int x, int y) { return (x < y) ? y : x; }

static int r_lowess_compare(double x, double y, r_lowess_boolean na_last) {
  int x_nan = isnan(x);
  int y_nan = isnan(y);

  if (x_nan && y_nan)
    return 0;
  if (x_nan)
    return na_last ? 1 : -1;
  if (y_nan)
    return na_last ? -1 : 1;
  if (x < y)
    return -1;
  if (x > y)
    return 1;
  return 0;
}

/* R's rPsort specialization for doubles. */
static void r_lowess_partial_sort(double *x, ptrdiff_t lo, ptrdiff_t hi,
                                  ptrdiff_t k) {
  ptrdiff_t left, right, i, j;
  double pivot, swap;

  for (left = lo, right = hi; left < right;) {
    pivot = x[k];
    for (i = left, j = right; i <= j;) {
      while (r_lowess_compare(x[i], pivot, R_LOWESS_TRUE) < 0)
        i++;
      while (r_lowess_compare(pivot, x[j], R_LOWESS_TRUE) < 0)
        j--;
      if (i <= j) {
        swap = x[i];
        x[i++] = x[j];
        x[j--] = swap;
      }
    }
    if (j < k)
      left = i;
    if (k < i)
      right = j;
  }
}

static void r_lowess_rpsort(double *x, int n, int k) {
  r_lowess_partial_sort(x, 0, n - 1, k);
}

static double r_lowess_square(double x) { return x * x; }

static double r_lowess_cube(double x) { return x * x * x; }

static void r_lowess_lowest(double *x, double *y, int n, double *xs, double *ys,
                            int nleft, int nright, double *w,
                            r_lowess_boolean userw, double *rw,
                            r_lowess_boolean *ok) {
  int nrt, j;
  double a, b, c, h, h1, h9, r, range;

  /* Preserve R's original one-based indexing and operation order. */
  x--;
  y--;
  w--;
  rw--;

  range = x[n] - x[1];
  h = r_lowess_fmax2(*xs - x[nleft], x[nright] - *xs);
  h9 = 0.999 * h;
  h1 = 0.001 * h;

  a = 0.;
  j = nleft;
  while (j <= n) {
    w[j] = 0.;
    r = fabs(x[j] - *xs);
    if (r <= h9) {
      if (r <= h1)
        w[j] = 1.;
      else
        w[j] = r_lowess_cube(1. - r_lowess_cube(r / h));
      if (userw)
        w[j] *= rw[j];
      a += w[j];
    } else if (x[j] > *xs) {
      break;
    }
    j++;
  }

  nrt = j - 1;
  if (a <= 0.) {
    *ok = R_LOWESS_FALSE;
    return;
  }

  *ok = R_LOWESS_TRUE;
  for (j = nleft; j <= nrt; j++)
    w[j] /= a;

  if (h > 0.) {
    a = 0.;
    for (j = nleft; j <= nrt; j++)
      a += w[j] * x[j];
    b = *xs - a;
    c = 0.;
    for (j = nleft; j <= nrt; j++)
      c += w[j] * r_lowess_square(x[j] - a);
    if (sqrt(c) > 0.001 * range) {
      b /= c;
      for (j = nleft; j <= nrt; j++)
        w[j] *= b * (x[j] - a) + 1.;
    }
  }

  *ys = 0.;
  for (j = nleft; j <= nrt; j++)
    *ys += w[j] * y[j];
}

static void r_lowess_clowess(double *x, double *y, int n, double f, int nsteps,
                             double delta, double *ys, double *rw,
                             double *res) {
  int i, iter, j, last, m1, m2, nleft, nright, ns;
  r_lowess_boolean ok;
  double alpha, c1, c9, cmad, cut, d1, d2, denom, r, sc;

  if (n < 2) {
    ys[0] = y[0];
    return;
  }

  x--;
  y--;
  ys--;

  ns = r_lowess_imax2(2, r_lowess_imin2(n, (int)(f * n + 1e-7)));
  iter = 1;
  while (iter <= nsteps + 1) {
    nleft = 1;
    nright = ns;
    last = 0;
    i = 1;

    for (;;) {
      if (nright < n) {
        d1 = x[i] - x[nleft];
        d2 = x[nright + 1] - x[i];
        if (d1 > d2) {
          nleft++;
          nright++;
          continue;
        }
      }

      r_lowess_lowest(&x[1], &y[1], n, &x[i], &ys[i], nleft, nright, res,
                      iter > 1, rw, &ok);
      if (!ok)
        ys[i] = y[i];

      if (last < i - 1) {
        denom = x[i] - x[last];
        for (j = last + 1; j < i; j++) {
          alpha = (x[j] - x[last]) / denom;
          ys[j] = alpha * ys[i] + (1. - alpha) * ys[last];
        }
      }

      last = i;
      cut = x[last] + delta;
      for (i = last + 1; i <= n; i++) {
        if (x[i] > cut)
          break;
        if (x[i] == x[last]) {
          ys[i] = ys[last];
          last = i;
        }
      }
      i = r_lowess_imax2(last + 1, i - 1);
      if (last >= n)
        break;
    }

    for (i = 0; i < n; i++)
      res[i] = y[i + 1] - ys[i + 1];

    sc = 0.;
    for (i = 0; i < n; i++)
      sc += fabs(res[i]);
    sc /= n;

    if (iter > nsteps)
      break;

    for (i = 0; i < n; i++)
      rw[i] = fabs(res[i]);

    m1 = n / 2;
    r_lowess_rpsort(rw, n, m1);
    if (n % 2 == 0) {
      m2 = n - m1 - 1;
      r_lowess_rpsort(rw, n, m2);
      cmad = 3. * (rw[m1] + rw[m2]);
    } else {
      cmad = 6. * rw[m1];
    }

    if (cmad < 1e-7 * sc)
      break;

    c9 = 0.999 * cmad;
    c1 = 0.001 * cmad;
    for (i = 0; i < n; i++) {
      r = fabs(res[i]);
      if (r <= c1)
        rw[i] = 1.;
      else if (r <= c9)
        rw[i] = r_lowess_square(1. - r_lowess_square(r / cmad));
      else
        rw[i] = 0.;
    }
    iter++;
  }
}

/*
 * Run R's LOWESS engine without linking to R.
 *
 * x must be sorted in ascending order. x, y, and ys must each contain n
 * doubles. Returns 0 on success and -1 for invalid arguments or allocation
 * failure.
 */
int r_lowess(double *x, double *y, int n, double f, int nsteps, double delta,
             double *ys) {
  double *rw, *res;

  if (x == NULL || y == NULL || ys == NULL || n < 1 || !isfinite(f) ||
      f <= 0. || nsteps < 0 || !isfinite(delta) || delta < 0.)
    return -1;

  rw = malloc((size_t)n * sizeof(*rw));
  res = malloc((size_t)n * sizeof(*res));
  if (rw == NULL || res == NULL) {
    free(rw);
    free(res);
    return -1;
  }

  r_lowess_clowess(x, y, n, f, nsteps, delta, ys, rw, res);
  free(rw);
  free(res);
  return 0;
}

/* Compatibility entry point retained for simple FFI/.C harnesses. */
void standalone_lowess(double *x, double *y, int n, double f, int nsteps,
                       double delta, double *ys) {
  (void)r_lowess(x, y, n, f, nsteps, delta, ys);
}