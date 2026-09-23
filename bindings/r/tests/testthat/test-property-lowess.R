#' @srrstats {G5.4, G5.4b} Reference comparison against base R's
#'   `stats::lowess`, generalized here to randomized inputs.
#' @srrstats {G5.10, G5.11} Extended/property-based tests, gated behind
#'   `skip_if_not_installed()` so they only run when `quickcheck` is
#'   available (not a hard package dependency) and are skipped on CRAN.
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
        if (length(unique(x)) != length(x)) {
            hedgehog::discard()
        }

        expect_matches_stats_lowess(
            x,
            y,
            fraction = fraction,
            iterations = iterations,
            tolerance = 1e-8
        )
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(5L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        # Includes the high end of the range where robustness weights can
        # collapse to a degenerate scale after enough reweighting passes.
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L
    )
})
