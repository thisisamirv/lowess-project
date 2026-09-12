#' @srrstats {G5.1, G5.5} Stored reference fixtures are generated under a
#'   fixed RNG seed recorded in tests/testthat/fixtures/PROVENANCE.txt.
#' @srrstats {G5.4c} Stored reference data: committed golden outputs in
#'   tests/testthat/fixtures/ are compared against a fresh fit.
#' @srrstats {G5.6, G5.6a} Parameter recovery against stored values within a
#'   1e-10 tolerance.

expect_matches_golden <- function(case) {
    fixture_dir <- testthat::test_path("fixtures")
    stored <- golden_read_csv(
        file.path(fixture_dir, paste0(case, ".csv"))
    )
    fresh <- golden_cases()[[case]]
    expect_equal(stored, fresh, tolerance = 1e-10)
}

test_that("batch default-config output matches the stored reference", {
    expect_matches_golden("batch_default")
})

test_that("batch interval/derivative output matches the stored reference", {
    expect_matches_golden("batch_intervals")
})

test_that("batch robustness weights match the stored reference", {
    expect_matches_golden("batch_robust")
})

test_that("streaming chunked output matches the stored reference", {
    expect_matches_golden("streaming_chunked")
})

test_that("online full-mode output matches the stored reference", {
    expect_matches_golden("online_full")
})
