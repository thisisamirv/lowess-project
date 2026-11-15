#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.8} Edge cases: min data, window > data.
#' @srrstats {RE4.0} Robustness iterations tested.
#' @srrstats {G5.6, G5.6a} Parameter recovery in online full mode.

ready_points <- function(results) {
    Filter(function(r) identical(r$status, "ready"), results)
}

test_that("OnlineLowess basic functionality works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        min_points = 10
    )
    results <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))

    expect_gt(length(ready_points(results)), 0)
})

test_that("OnlineLowess window capacity works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_small <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 15
    )
    results_small <- lapply(
        seq_along(x),
        function(i) add_point(ol_small, x[[i]], y[[i]])
    )

    ol_large <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 50
    )
    results_large <- lapply(
        seq_along(x),
        function(i) add_point(ol_large, x[[i]], y[[i]])
    )

    expect_gt(length(ready_points(results_small)), 0)
    expect_gt(length(ready_points(results_large)), 0)
})

test_that("OnlineLowess min_points parameter works", {
    set.seed(42)
    x <- as.double(1:50)
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        min_points = 5
    )
    results <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))

    expect_identical(results[[1]]$status, "filling")
    expect_null(results[[1]]$point)
    expect_gt(length(ready_points(results)), 0)
})

test_that("OnlineLowess update modes work", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_full <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        update_mode = "full"
    )
    results_full <- lapply(
        seq_along(x),
        function(i) add_point(ol_full, x[[i]], y[[i]])
    )

    ol_incr <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        update_mode = "incremental"
    )
    results_incr <- lapply(
        seq_along(x),
        function(i) add_point(ol_incr, x[[i]], y[[i]])
    )

    expect_gt(length(ready_points(results_full)), 0)
    expect_gt(length(ready_points(results_incr)), 0)
})

test_that("OnlineLowess handles edge cases", {
    # Minimum data points
    x <- as.double(1:10)
    y <- as.double(1:10)
    ol <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 5,
        min_points = 3
    )
    results <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))
    expect_gt(length(ready_points(results)), 0)

    # Window larger than data
    ol2 <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 20,
        min_points = 3
    )
    results2 <- lapply(
        seq_along(x),
        function(i) add_point(ol2, x[[i]], y[[i]])
    )
    expect_gt(length(ready_points(results2)), 0)
})

test_that("OnlineLowess robustness works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)
    y[50] <- y[50] + 5 # Add outlier

    ol_no_robust <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        iterations = 0
    )
    results_no_robust <- lapply(
        seq_along(x),
        function(i) add_point(ol_no_robust, x[[i]], y[[i]])
    )

    ol_robust <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        iterations = 3,
        update_mode = "full"
    )
    results_robust <- lapply(
        seq_along(x),
        function(i) add_point(ol_robust, x[[i]], y[[i]])
    )

    expect_gt(length(ready_points(results_no_robust)), 0)
    expect_gt(length(ready_points(results_robust)), 0)
})

test_that("G5.6 OnlineLowess full mode recovers a noiseless signal", {
    x <- as.double(seq(0, 10, length.out = 40))
    y <- as.double(2 * x + 1)

    # update_mode = "full" with iterations > 0 recovers the noiseless
    # expected smoothed values exactly.
    full <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 12L,
        min_points = 4L,
        update_mode = "full",
        iterations = 2L
    )
    out_full <- lapply(
        seq_along(x),
        function(i) add_point(full, x[[i]], y[[i]])
    )
    yhat_full <- vapply(
        out_full,
        function(r) {
            if (identical(r$status, "ready")) r$point$y else NA_real_
        },
        numeric(1)
    )
    valid <- !is.na(yhat_full)
    expect_true(any(valid))
    expect_equal(yhat_full[valid], as.double(y[valid]), tolerance = 1e-12)

    # update_mode = "incremental" with iterations > 0 is rejected at build
    # time rather than silently ignoring the robustness setting.
    expect_error(
        OnlineLowess(
            fraction = 0.3,
            window_capacity = 12L,
            min_points = 4L,
            update_mode = "incremental",
            iterations = 2L
        ),
        "iterations > 0 requires update_mode"
    )

    # update_mode = "incremental" with iterations = 0 is the correctly
    # specified non-robust path and runs without error.
    incr <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 12L,
        min_points = 4L,
        update_mode = "incremental",
        iterations = 0L
    )
    expect_no_error(
        lapply(seq_along(x), function(i) add_point(incr, x[[i]], y[[i]]))
    )
})

test_that("OnlineLowess missing = \"drop\" distinguishes drop from filling", {
    ol <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 10,
        min_points = 3,
        missing = "drop"
    )

    filling <- add_point(ol, 0.0, 0.0)
    expect_identical(filling$status, "filling")
    expect_null(filling$point)

    dropped <- add_point(ol, 1.0, NaN)
    expect_identical(dropped$status, "dropped")
    expect_null(dropped$point)

    still_filling <- add_point(ol, 2.0, 4.0)
    expect_identical(still_filling$status, "filling")
    expect_null(still_filling$point)
})

test_that("OnlineLowess return_se/ci/pi requires full mode", {
    expect_error(
        OnlineLowess(
            fraction = 0.5,
            window_capacity = 10,
            min_points = 3,
            outputs = "se"
        )
    )
})

test_that("OnlineLowess return_se/ci/pi work", {
    set.seed(42)
    x <- as.double(1:30)
    y <- sin(x / 10) + rnorm(30, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 10,
        min_points = 3,
        update_mode = "full",
        outputs = "se",
        intervals = intervals_opts(confidence = 0.95, prediction = 0.95)
    )
    results <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))
    ready <- ready_points(results)

    expect_gt(length(ready), 0)
    last <- ready[[length(ready)]]$point
    expect_false(is.null(last$standard_error))
    expect_false(is.null(last$confidence_lower))
    expect_false(is.null(last$confidence_upper))
    expect_false(is.null(last$prediction_lower))
    expect_false(is.null(last$prediction_upper))
})

test_that("OnlineLowess bootstrap intervals are reproducible with a seed", {
    set.seed(42)
    x <- as.double(1:30)
    y <- sin(x / 10) + rnorm(30, sd = 0.1)

    run <- function() {
        ol <- OnlineLowess(
            fraction = 0.5,
            window_capacity = 10,
            min_points = 3,
            update_mode = "full",
            intervals = intervals_opts(confidence = 0.95, bootstrap = 30),
            seed = 5
        )
        res <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))
        ready_points(res)
    }
    r1 <- run()
    r2 <- run()
    last1 <- r1[[length(r1)]]$point
    last2 <- r2[[length(r2)]]$point
    expect_false(is.null(last1$confidence_lower))
    expect_identical(last1$confidence_lower, last2$confidence_lower)
})

test_that("add_point rejects non-scalar x/y", {
    ol <- OnlineLowess(fraction = 0.5, window_capacity = 10)
    expect_error(add_point(ol, c(1, 2), 3), "single numeric")
    expect_error(add_point(ol, 1, c(2, 3)), "single numeric")
})
