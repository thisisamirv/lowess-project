#' @srrstats {G5.4, G5.4b} Reference comparison against base R's
#'   `stats::lowess`, generalized here to randomized inputs.
#' @srrstats {G5.10} Extended/property-based test, gated behind
#'   `skip_if_not_installed()` so it only runs when `quickcheck` is
#'   available (not a hard package dependency) and is skipped on CRAN.
#'
# Property-based regression test: fuzzes x/y/fraction/iterations and checks
# that this package's output matches `stats::lowess` (via
# `expect_matches_stats_lowess()` in helper-validation.R), pinning R's
# defaults (`boundary_policy = "noboundary"`, `scaling_method = "mar"`).
#
# This complements the fixed scenarios in test-validation.R: a fixed
# regression divergence at high robustness-iteration counts (see NEWS/
# CHANGELOG) was only found by randomized fuzzing over many (x, y,
# fraction, iterations) combinations, not by any single hand-picked case.
#' @noRd
qc <- function(name) {
    getExportedValue("quickcheck", name)
}

usable_x <- function(x, min_length = 5L) {
    length(x) >= min_length && anyDuplicated(x) == 0L
}

test_that("matches stats::lowess for randomized inputs (property-based)", {
    testthat::skip_if_not_installed("quickcheck")
    testthat::skip_on_cran()

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
            tolerance = 1e-5
        ))
    }

    qc("for_all")(
        xy = qc("equal_length")(
            qc("double_bounded")(-100, 100, len = c(5L, 40L)),
            qc("double_bounded")(-100, 100, len = c(5L, 40L))
        ),
        fraction = qc("double_bounded")(0.05, 1.0, len = 1L),
        # Broad fuzzing covers enough passes to expose branch differences;
        # long-run floating-point cycles are pinned by fixed regressions.
        iterations = qc("integer_bounded")(0L, 12L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches stats::lowess for randomized sorted output", {
    testthat::skip_if_not_installed("quickcheck")
    testthat::skip_on_cran()

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
            tolerance = 1e-5
        ))
    }

    qc("for_all")(
        xy = qc("equal_length")(
            qc("double_bounded")(-100, 100, len = c(5L, 40L)),
            qc("double_bounded")(-100, 100, len = c(5L, 40L))
        ),
        fraction = qc("double_bounded")(0.05, 1.0, len = 1L),
        iterations = qc("integer_bounded")(0L, 12L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches initial stats::lowess fits for sparse one-spike responses", {
    testthat::skip_if_not_installed("quickcheck")
    testthat::skip_on_cran()

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
            tolerance = 1e-8
        ))
    }

    qc("for_all")(
        x = qc("double_bounded")(-100, 100, len = c(5L, 40L)),
        spike_position = qc("double_bounded")(0, 1, len = 1L),
        spike_magnitude = qc("double_bounded")(1e-4, 100, len = 1L),
        spike_negative = qc("logical_")(len = 1L),
        fraction = qc("double_bounded")(0.05, 1.0, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})
