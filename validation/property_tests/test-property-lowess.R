#' @srrstats {G5.4, G5.4b} Reference comparison against base R's
#'   `stats::lowess`, generalized here to randomized inputs.
#' @srrstats {G5.10} Property-based tests run in the standard suite.
#'   `quickcheck` is installed by `make validate`, not a package dependency.
#' @noRd

# Property-based regression test: fuzzes x/y/fraction/iterations and checks
# that this package's output matches `stats::lowess` (via
# `expect_matches_stats_lowess()` in helper-validation.R), pinning R's
# comparison settings (`boundary_policy = "noboundary"`,
# `scaling_method = "mar"`, `zero_weight_fallback = "return_original"`).
# The package default remains `"use_local_mean"`; these comparisons opt into
# R's fallback explicitly.
#
# This complements the fixed scenarios in test-validation.R: randomized
# comparisons can expose divergences at iteration counts not covered by any
# single hand-picked case.

# quickcheck shrinks below the declared minimum length, so the two-point
# minimum that both `stats::lowess` and this package require is enforced here.
usable_x <- function(x, min_length = 2L) {
    length(x) >= min_length
}

test_that("cmad noise-floor guard handles unsorted x", {
    x <- c(
        18.696686858311296,
        5.791397113353014,
        -61.549769470468163,
        71.320148417726159
    )
    y <- c(-11.987563783981185, 0, 0, 0)
    fraction <- 0.964674779062625
    iterations <- 168L
    ord <- order(x)

    expect_true(cmad_is_noise_floor(x[ord], y[ord], fraction, iterations))
    expect_identical(
        cmad_is_noise_floor(x, y, fraction, iterations),
        cmad_is_noise_floor(x[ord], y[ord], fraction, iterations)
    )
})

test_that("matches stats::lowess for randomized inputs (property-based)", {
    property <- function(xy, fraction, iterations) {
        if (!usable_x(xy[[1]])) {
            return(expect_true(TRUE))
        }
        expect_true(check_stats_lowess(
            xy[[1]],
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches stats::lowess for randomized sorted output", {
    property <- function(xy, fraction, iterations) {
        if (!usable_x(xy[[1]])) {
            return(expect_true(TRUE))
        }
        expect_true(check_stats_lowess(
            xy[[1]],
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 25L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

# Continuous draws practically never collide, so ties need their own generator.
# `stats::lowess` copies one fitted value across a run of tied x-values instead
# of refitting each point, and collapsing the draw onto a handful of levels
# exercises that path (including the all-tied case, where every x is equal).
test_that("matches stats::lowess for tied x-values (property-based)", {
    property <- function(xy, levels, fraction, iterations) {
        if (!usable_x(xy[[1]])) {
            return(expect_true(TRUE))
        }
        x <- round(xy[[1]] / (200 / levels))
        expect_true(check_stats_lowess(
            x,
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        levels = quickcheck::integer_bounded(1L, 8L, len = 1L),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches initial stats::lowess fits for sparse one-spike responses", {
    property <- function(
        x,
        spike_position,
        spike_magnitude,
        spike_negative,
        fraction,
        iterations
    ) {
        if (!usable_x(x)) {
            return(expect_true(TRUE))
        }

        spike_index <- min(
            length(x),
            floor(spike_position * length(x)) + 1L
        )
        spike_value <- if (spike_negative) -spike_magnitude else spike_magnitude
        y <- numeric(length(x))
        y[spike_index] <- spike_value

        expect_true(check_stats_lowess(
            x,
            y,
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        x = quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
        spike_position = quickcheck::double_bounded(0, 1, len = 1L),
        spike_magnitude = quickcheck::double_bounded(1e-4, 100, len = 1L),
        spike_negative = quickcheck::logical_(len = 1L),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})
