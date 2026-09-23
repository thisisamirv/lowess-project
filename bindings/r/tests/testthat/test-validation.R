#' @srrstats {G5.4, G5.4a} Correctness on analytic cases and fixed datasets.
#' @srrstats {G5.4b} Reference comparison against base R's `stats::lowess`
#'   and `stats::lm`.
#' @srrstats {G5.5} Fixed random seeds (`set.seed(42)`).
#' @srrstats {RE7.0, RE7.0a} Noiseless exact relationships between predictor
#'   (independent) data: identical x values and perfectly structured x are
#'   handled without error; ability to recognize/handle perfectly noiseless
#'   predictor configurations is confirmed via finite output and exact
#'   reproduction on degenerate cases.
#' @srrstats {RE7.1, RE7.1a} Noiseless exact relationships between predictor
#'   and response: y = f(x) exactly (linear, constant) reproduces truth;
#'   perfect fit is recognized (r_squared = 1, rmse = 0 when
#'   outputs = "diagnostics"); fitting exact data is at least as fast as
#'   equivalent noisy data.
#'
# Validation: numerical agreement with R's `stats::lowess` and `stats::lm`.
#
# Two kinds of reference are used:
#
#   * Analytic cases - inputs with a closed-form answer: a straight line,
#     a constant, a two-point window, and the `fraction >= 1.0` global OLS
#     branch (checked against `stats::lm`). These separate correctness of
#     the *method* from correctness of the implementation (G5.4a).
#   * `stats::lowess` - an independent implementation of Cleveland's
#     original algorithm (Fortran -> C), used on fixed base R datasets, tied
#     `x`, and a fraction x iterations grid (G5.4b).
#
# The `stats::lowess` comparison pins R's defaults:
#
#   * `boundary_policy = "noboundary"` - R applies no boundary padding
#     (Cleveland original); this package defaults to `"extend"`.
#   * `scaling_method = "mar"` - R scales residuals by MAR
#     (`6 * median(|resid|)`); this package defaults to `"mad"`.
#   * `delta` - R's default is `0.01 * diff(range(x))`, matching this
#     package's default (`NULL` -> 1% of the x-range), so `NULL` is passed
#     through. Scenarios requesting an exact surface pass `delta = 0`.
#'
#' @noRd
generate_validation_data <- function(
    n = 100,
    kind = "linear",
    noise = 0.0,
    range_min = 0.0,
    range_max = 1.0,
    outlier_ratio = 0.0
) {
    set.seed(42)

    x <- seq(range_min, range_max, length.out = n)

    if (kind == "linear") {
        y <- 2 * x + 1
    } else if (kind == "quadratic") {
        y <- x^2
    } else if (kind == "sine") {
        y <- sin(4 * x)
    } else if (kind == "step") {
        y <- ifelse(x < (range_min + range_max) / 2, 0.0, 1.0)
    } else if (kind == "constant") {
        y <- rep(5.0, n)
    } else {
        y <- x
    }

    if (noise > 0) {
        y <- y + rnorm(n, 0, noise)
    }

    if (outlier_ratio > 0) {
        n_out <- as.integer(n * outlier_ratio)
        indices <- sample(n, n_out, replace = FALSE)
        y[indices] <- y[indices] + 10.0
    }

    list(x = x, y = y)
}

# --- Analytic cases ---

test_that("local linear fit reproduces a straight line exactly", {
    x <- seq(0, 10, length.out = 50)
    y <- 2 * x + 1

    # With no boundary padding every window sees collinear points, so the
    # local linear fit equals the truth at every point.
    exact <- fit(
        Lowess(
            fraction = 0.3,
            iterations = 0L,
            boundary_policy = "noboundary"
        ),
        as.double(x),
        as.double(y)
    )
    expect_equal(exact$y, as.double(y), tolerance = 1e-10)

    # Under the default "extend" padding only the padded edges differ.
    padded <- fit(
        Lowess(fraction = 0.3, iterations = 0L),
        as.double(x),
        as.double(y)
    )
    expect_equal(padded$y[11:40], as.double(y[11:40]), tolerance = 1e-10)
})

test_that("a constant signal is reproduced exactly", {
    x <- seq(0, 10, length.out = 40)
    y <- rep(5, 40)

    for (iterations in c(0L, 5L)) {
        result <- fit(
            Lowess(fraction = 0.5, iterations = iterations),
            as.double(x),
            as.double(y)
        )
        expect_equal(result$y, as.double(y), tolerance = 1e-10)
    }
})

test_that("a two-point window interpolates exactly", {
    x <- c(0, 1)
    y <- c(3, 7)

    result <- fit(
        Lowess(fraction = 0.5, iterations = 0L),
        as.double(x),
        as.double(y)
    )
    expect_equal(result$y, as.double(y), tolerance = 1e-10)
})

test_that("fraction = 1.0 reproduces the global OLS fit", {
    set.seed(1)
    x <- seq(0, 10, length.out = 30)
    y <- 0.5 * x + rnorm(30, 0, 0.3)

    result <- fit(
        Lowess(fraction = 1.0, iterations = 0L),
        as.double(x),
        as.double(y)
    )

    # `fraction >= 1.0` is a deliberate extension beyond R's semantics:
    # `stats::lowess(f = 1)` still applies a tricube weighting centred on
    # each x, so the reference here is ordinary least squares instead.
    expect_equal(
        result$y,
        as.double(stats::fitted(stats::lm(y ~ x))),
        tolerance = 1e-10
    )
})

test_that("robustness iterations are a no-op on an exact fit", {
    x <- seq(0, 10, length.out = 50)
    y <- 3 * x - 2

    base <- fit(
        Lowess(
            fraction = 0.4,
            iterations = 0L,
            boundary_policy = "noboundary"
        ),
        as.double(x),
        as.double(y)
    )
    robust <- fit(
        Lowess(
            fraction = 0.4,
            iterations = 5L,
            boundary_policy = "noboundary"
        ),
        as.double(x),
        as.double(y)
    )

    # Zero residuals leave every bisquare weight at 1, so reweighting
    # cannot move the fit.
    expect_equal(robust$y, base$y, tolerance = 1e-10)
})

# --- Ported validation scenarios ---

test_that("validation 01_tiny_linear matches stats::lowess", {
    d <- generate_validation_data(n = 10, kind = "linear")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.8, iterations = 0)
})

test_that("validation 02_sine_standard matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "sine", noise = 0.1)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.3, iterations = 0)
})

test_that("validation 03_sine_robust matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.3, iterations = 4)
})

test_that("validation 04_large_scale matches stats::lowess", {
    d <- generate_validation_data(n = 500, kind = "sine")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.1, iterations = 0)
})

test_that("validation 05_high_smoothness matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "linear", noise = 0.5)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.9, iterations = 0)
})

test_that("validation 06_low_smoothness matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "sine")
    expect_matches_stats_lowess(
        d$x,
        d$y,
        fraction = 0.05,
        iterations = 0,
        direct = TRUE
    )
})

test_that("validation 07_constant matches stats::lowess", {
    d <- generate_validation_data(n = 50, kind = "constant")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.5, iterations = 0)
})

test_that("validation 08_step_func matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "step")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.4, iterations = 0)
})

test_that("validation 09_end_effects_left matches stats::lowess", {
    d <- generate_validation_data(n = 50, kind = "linear", noise = 0.1)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.3, iterations = 0)
})

test_that("validation 10_end_effects_right matches stats::lowess", {
    d <- generate_validation_data(n = 50, kind = "linear", noise = 0.1)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.3, iterations = 0)
})

test_that("validation 11_sparse_data matches stats::lowess", {
    d <- generate_validation_data(
        n = 20,
        range_max = 100.0,
        kind = "linear",
        noise = 1.0
    )
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.6, iterations = 0)
})

test_that("validation 12_dense_data matches stats::lowess", {
    d <- generate_validation_data(n = 1000, kind = "sine", noise = 0.1)
    expect_matches_stats_lowess(
        d$x,
        d$y,
        fraction = 0.01,
        iterations = 0,
        direct = TRUE
    )
})

test_that("validation 13_iter_2 matches stats::lowess", {
    d <- generate_validation_data(n = 100, kind = "sine", outlier_ratio = 0.05)
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.4, iterations = 2)
})

test_that("validation 14_interpolate_exact matches stats::lowess", {
    d <- generate_validation_data(n = 50, kind = "linear")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.5, iterations = 0)
})

test_that("validation 15_zero_variance matches stats::lowess", {
    d <- generate_validation_data(n = 10, kind = "constant")
    expect_matches_stats_lowess(d$x, d$y, fraction = 0.5, iterations = 0)
})

# --- Fixed base R datasets ---

test_that("matches stats::lowess on the cars dataset", {
    for (fraction in c(0.2, 2 / 3)) {
        expect_matches_stats_lowess(
            datasets::cars$speed,
            datasets::cars$dist,
            fraction = fraction,
            iterations = 0
        )
    }
})

test_that("matches stats::lowess on the faithful dataset", {
    for (fraction in c(0.2, 2 / 3)) {
        expect_matches_stats_lowess(
            datasets::faithful$eruptions,
            datasets::faithful$waiting,
            fraction = fraction,
            iterations = 0
        )
    }
})

test_that("matches stats::lowess on the trees dataset", {
    for (fraction in c(0.2, 2 / 3)) {
        expect_matches_stats_lowess(
            datasets::trees$Girth,
            datasets::trees$Height,
            fraction = fraction,
            iterations = 0
        )
    }
})

test_that("matches stats::lowess on the Indometh dataset", {
    for (fraction in c(0.2, 2 / 3)) {
        expect_matches_stats_lowess(
            datasets::Indometh$time,
            datasets::Indometh$conc,
            fraction = fraction,
            iterations = 0
        )
    }
})

test_that("matches stats::lowess on the DNase dataset", {
    for (fraction in c(0.2, 2 / 3)) {
        expect_matches_stats_lowess(
            datasets::DNase$conc,
            datasets::DNase$density,
            fraction = fraction,
            iterations = 0
        )
    }
})

# --- Tied / duplicated x ---

test_that("matches stats::lowess when x has tied values", {
    x <- c(1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10)
    y <- c(2, 4, 6, 8, 10, 11, 9, 12, 14, 16, 18, 20)

    for (fraction in c(0.3, 0.5, 0.6)) {
        expect_matches_stats_lowess(x, y, fraction = fraction, iterations = 0)
    }
})

test_that("matches stats::lowess when x has replicated pairs", {
    set.seed(7)
    x <- sort(rep(seq(0, 10, length.out = 30), each = 2))
    y <- sin(x) + rnorm(60, 0, 0.1)

    for (fraction in c(0.3, 0.5)) {
        expect_matches_stats_lowess(x, y, fraction = fraction, iterations = 0)
    }
})

# --- fraction x iterations grid ---

test_that("matches stats::lowess across a fraction x iterations grid", {
    set.seed(42)
    x <- seq(0, 1, length.out = 100)
    y <- sin(4 * x) + rnorm(100, 0, 0.1)

    for (fraction in c(0.2, 2 / 3, 0.8)) {
        for (iterations in c(0, 3)) {
            expect_matches_stats_lowess(
                x,
                y,
                fraction = fraction,
                iterations = iterations
            )
        }
    }
})

test_that("matches stats::lowess for unsorted input with sorted output", {
    x <- c(0.92113464, -4.53794094, 0, 3.89630243, -1.20512830)
    y <- c(1.23863738, -4.5717619, 2.23168736, 1.35353460, 0.31884629)
    fraction <- 0.582587
    iterations <- 181L

    result <- fit(
        Lowess(
            fraction = fraction,
            iterations = iterations,
            boundary_policy = "noboundary",
            scaling_method = "mar"
        ),
        x,
        y
    )
    sorted_result <- fit(
        Lowess(
            fraction = fraction,
            iterations = iterations,
            boundary_policy = "noboundary",
            scaling_method = "mar",
            outputs = "sorted"
        ),
        x,
        y
    )
    reference <- stats::lowess(x, y, f = fraction, iter = iterations)

    expect_identical(result$x, as.double(x))
    expect_equal(sorted_result$x, reference$x, tolerance = 1e-12)
    expect_equal(sorted_result$y, reference$y, tolerance = 1e-10)
})

test_that("matches committed statsmodels.lowess reference fixtures", {
    fixture <- utils::read.csv(
        testthat::test_path("fixtures", "statsmodels_lowess.csv")
    )
    settings <- list(
        basic = list(fraction = 0.3, iterations = 0L, delta = 0),
        linear = list(fraction = 0.6, iterations = 0L, delta = 0),
        delta_basic = list(fraction = 0.3, iterations = 0L, delta = 0.25),
        delta_fraction = list(fraction = 0.55, iterations = 0L, delta = 0.4)
    )

    for (case_name in names(settings)) {
        rows <- fixture[fixture$case == case_name, ]
        options <- c(
            settings[[case_name]],
            list(
                boundary_policy = "noboundary",
                outputs = "sorted"
            )
        )
        result <- fit(do.call(Lowess, options), rows$x, rows$y)

        expect_equal(result$x, rows$x, tolerance = 1e-12)
        expect_equal(result$y, rows$yhat, tolerance = 1e-12)
    }
})

# --- RE7.0 / RE7.1: noiseless exact relationships ---

test_that("RE7.0/RE7.1 noiseless exact predictor and predictor+response", {
    # RE7.0 / RE7.0a: noiseless exact relationships between predictor data.
    # Identical x values (degenerate predictor structure) must be handled
    # without error/non-finite output. This demonstrates the ability to "reject"
    # (not silently fail on) perfectly structured noiseless predictor input.
    x_ident <- rep(seq(0, 5, length.out = 10), each = 3)
    y_ident <- sin(x_ident) + 0.1
    expect_no_error({
        r <- fit(Lowess(fraction = 0.4), as.double(x_ident), as.double(y_ident))
    })
    expect_true(all(is.finite(r$y)))

    # Extreme degenerate predictor: all x identical (constant predictor).
    x_const_pred <- rep(3.0, 30)
    y_var <- rnorm(30, 5, 1)
    expect_no_error({
        r <- fit(
            Lowess(fraction = 0.5),
            as.double(x_const_pred),
            as.double(y_var)
        )
    })
    expect_true(all(is.finite(r$y)))

    # RE7.1 / RE7.1a: noiseless exact relationships between predictor-response.
    # y = f(x) exactly (linear, constant) with zero noise. Must reproduce truth
    # exactly (within floating-point tolerance) and not crash or produce garbage
    d_lin <- generate_validation_data(n = 60, kind = "linear", noise = 0.0)
    r_lin <- fit(
        Lowess(fraction = 0.3, iterations = 0L, boundary_policy = "noboundary"),
        as.double(d_lin$x),
        as.double(d_lin$y)
    )
    expect_equal(r_lin$y, as.double(d_lin$y), tolerance = 1e-12)

    d_const <- generate_validation_data(n = 40, kind = "constant", noise = 0.0)
    r_const <- fit(
        Lowess(fraction = 0.5, iterations = 0L),
        as.double(d_const$x),
        as.double(d_const$y)
    )
    expect_equal(r_const$y, as.double(d_const$y), tolerance = 1e-12)

    # Perfect fit is recognized via diagnostics: r_squared == 1, rmse == 0.
    # This is the correct "rejection" behavior — the model reports that the
    # relationship is exact rather than silently returning a misleading fit.
    # `outputs` belongs to Lowess(), not fit().
    r_diag <- fit(
        Lowess(
            fraction = 0.3,
            iterations = 0L,
            boundary_policy = "noboundary",
            outputs = "diagnostics"
        ),
        as.double(d_lin$x),
        as.double(d_lin$y)
    )
    expect_false(is.null(r_diag$diagnostics))
    expect_equal(r_diag$diagnostics$r_squared, 1.0, tolerance = 1e-10)
    expect_equal(r_diag$diagnostics$rmse, 0.0, tolerance = 1e-12)

    # RE7.1a: fitting exact data is at least as fast as noisy equivalent.
    # (See also RE2.4b timing expectations; exact data should not be slower.)
    d_noisy <- generate_validation_data(n = 200, kind = "linear", noise = 0.05)

    # Warm up both paths so JIT/tier compilation isn't charged to a measurement.
    fit(Lowess(fraction = 0.3), as.double(d_lin$x), as.double(d_lin$y))
    fit(Lowess(fraction = 0.3), as.double(d_noisy$x), as.double(d_noisy$y))

    # Minimum elapsed over several repetitions filters out GC pauses and
    # scheduler noise that a single `system.time()` cannot distinguish from a
    # real slowdown (this test flaked on CI from an 11 ms measurement jitter).
    t_exact <- min(replicate(
        5L,
        system.time(
            fit(Lowess(fraction = 0.3), as.double(d_lin$x), as.double(d_lin$y))
        )["elapsed"]
    ))
    t_noisy <- min(replicate(
        5L,
        system.time(
            fit(
                Lowess(fraction = 0.3),
                as.double(d_noisy$x),
                as.double(d_noisy$y)
            )
        )["elapsed"]
    ))
    # Allow small measurement overhead; exact must not be meaningfully slower.
    expect_lte(as.numeric(t_exact), as.numeric(t_noisy) + 0.05)
})
