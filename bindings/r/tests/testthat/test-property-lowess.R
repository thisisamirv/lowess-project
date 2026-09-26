#' @srrstats {G5.4, G5.4b} Reference comparison against base R's
#'   `stats::lowess`, generalized here to randomized inputs.
#' @srrstats {G5.10} Property-based tests run in the standard suite.
#'   `quickcheck` is a Suggests test dependency, not a package dependency.
#' @noRd

# Property-based regression test: fuzzes x/y/fraction/iterations and checks
# that this package's output matches `stats::lowess` (via
# `expect_matches_stats_lowess()` in helper-validation.R), pinning R's
# comparison settings (`boundary_policy = "noboundary"`,
# `scaling_method = "mar"`, `zero_weight_fallback = "return_original"`).
# The package default remains `"use_local_mean"`; these comparisons opt into
# R's fallback explicitly.
#
# This complements the fixed scenarios in test-validation.R: a fixed
# regression divergence at high robustness-iteration counts (see NEWS/
# CHANGELOG) was only found by randomized fuzzing over many (x, y,
# fraction, iterations) combinations, not by any single hand-picked case.
usable_x <- function(x, min_length = 5L) {
    length(x) >= min_length && anyDuplicated(x) == 0L
}

test_that("matches stats::lowess for randomized inputs (property-based)", {
    property <- function(xy, fraction, iterations) {
        x <- xy[[1]]
        y <- xy[[2]]

        # Duplicate/near-duplicate x values exercise a separate tie-handling
        # code path already covered by dedicated fixed cases in
        # test-validation.R ("... when x has tied values" / "... replicated
        # pairs"); keep this fuzz test focused on the well-conditioned,
        # distinct-x class of input from the original bug report by
        # discarding ties and letting quickcheck draw a new case.
        if (!usable_x(x)) {
            return(expect_true(TRUE))
        }
        expect_true(check_stats_lowess(
            x,
            y,
            fraction = fraction,
            iterations = iterations,
            zero_weight_fallback = "return_original",
            tolerance = 1e-5
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        # Broad fuzzing covers enough passes to expose branch differences;
        # long-run floating-point cycles are pinned by fixed regressions.
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches stats::lowess for randomized sorted output", {
    property <- function(xy, fraction, iterations) {
        x <- xy[[1]]
        y <- xy[[2]]

        if (!usable_x(x)) {
            return(expect_true(TRUE))
        }
        expect_true(check_stats_lowess(
            x,
            y,
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-5
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
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
        fraction
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
            iterations = 0L,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-8
        ))
    }

    quickcheck::for_all(
        x = quickcheck::double_bounded(-100, 100, len = c(5L, 40L)),
        spike_position = quickcheck::double_bounded(0, 1, len = 1L),
        spike_magnitude = quickcheck::double_bounded(1e-4, 100, len = 1L),
        spike_negative = quickcheck::logical_(len = 1L),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})
