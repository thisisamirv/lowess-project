#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.8} Edge cases: min data, window > data.
#' @srrstats {RE4.0} Robustness iterations tested.
#' @srrstats {G5.6, G5.6a} Parameter recovery in online full mode.
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

    expect_false(all(vapply(results, is.null, logical(1))))
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

    expect_false(all(vapply(results_small, is.null, logical(1))))
    expect_false(all(vapply(results_large, is.null, logical(1))))
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

    # Results before min_points should be NULL
    expect_null(results[[1]])
    # At least some results should be non-NULL
    expect_false(all(vapply(results, is.null, logical(1))))
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

    expect_false(all(vapply(results_full, is.null, logical(1))))
    expect_false(all(vapply(results_incr, is.null, logical(1))))
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
    expect_false(all(vapply(results, is.null, logical(1))))

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
    expect_false(all(vapply(results2, is.null, logical(1))))
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

    expect_false(all(vapply(results_no_robust, is.null, logical(1))))
    expect_false(all(vapply(results_robust, is.null, logical(1))))
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
        function(r) if (is.null(r)) NA_real_ else r$y,
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

test_that("OnlineLowess missing = \"drop\" ignores non-finite point", {
    ol <- OnlineLowess(
        fraction = 0.5,
        window_capacity = 10,
        missing = "drop"
    )

    result <- add_point(ol, 1.0, NaN)
    expect_null(result)
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
        confidence_intervals = 0.95,
        prediction_intervals = 0.95
    )
    results <- lapply(seq_along(x), function(i) add_point(ol, x[[i]], y[[i]]))
    non_null <- Filter(Negate(is.null), results)

    expect_gt(length(non_null), 0)
    last <- non_null[[length(non_null)]]
    expect_false(is.null(last$standard_error))
    expect_false(is.null(last$confidence_lower))
    expect_false(is.null(last$confidence_upper))
    expect_false(is.null(last$prediction_lower))
    expect_false(is.null(last$prediction_upper))
})
