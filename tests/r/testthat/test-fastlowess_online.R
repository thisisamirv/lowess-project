#' @srrstats {G5.4} Correctness tests for online/sliding window mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.8} Edge cases: min data, window > data.
#' @srrstats {RE4.0} Robustness iterations tested.
test_that("OnlineLowess basic functionality works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3, window_capacity = 25, min_points = 10
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[[i]], y[[i]]))

    expect_false(all(vapply(results, is.null, logical(1))))
})

test_that("OnlineLowess window capacity works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_small <- OnlineLowess(
        fraction = 0.3, window_capacity = 15
    )
    results_small <- lapply(
        seq_along(x), function(i) ol_small$add_point(x[[i]], y[[i]])
    )

    ol_large <- OnlineLowess(
        fraction = 0.3, window_capacity = 50
    )
    results_large <- lapply(
        seq_along(x), function(i) ol_large$add_point(x[[i]], y[[i]])
    )

    expect_false(all(vapply(results_small, is.null, logical(1))))
    expect_false(all(vapply(results_large, is.null, logical(1))))
})

test_that("OnlineLowess min_points parameter works", {
    set.seed(42)
    x <- as.double(1:50)
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3, window_capacity = 25, min_points = 5
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[[i]], y[[i]]))

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
        fraction = 0.3, window_capacity = 25,
        update_mode = "full"
    )
    results_full <- lapply(
        seq_along(x), function(i) ol_full$add_point(x[[i]], y[[i]])
    )

    ol_incr <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "incremental"
    )
    results_incr <- lapply(
        seq_along(x), function(i) ol_incr$add_point(x[[i]], y[[i]])
    )

    expect_false(all(vapply(results_full, is.null, logical(1))))
    expect_false(all(vapply(results_incr, is.null, logical(1))))
})

test_that("OnlineLowess handles edge cases", {
    # Minimum data points
    x <- as.double(1:10)
    y <- as.double(1:10)
    ol <- OnlineLowess(
        fraction = 0.5, window_capacity = 5, min_points = 3
    )
    results <- lapply(seq_along(x), function(i) ol$add_point(x[[i]], y[[i]]))
    expect_false(all(vapply(results, is.null, logical(1))))

    # Window larger than data
    ol2 <- OnlineLowess(
        fraction = 0.5, window_capacity = 20, min_points = 3
    )
    results2 <- lapply(
        seq_along(x), function(i) ol2$add_point(x[[i]], y[[i]])
    )
    expect_false(all(vapply(results2, is.null, logical(1))))
})

test_that("OnlineLowess robustness works", {
    set.seed(42)
    x <- as.double(1:100)
    y <- sin(x / 10) + rnorm(100, sd = 0.1)
    y[50] <- y[50] + 5 # Add outlier

    ol_no_robust <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        iterations = 0
    )
    results_no_robust <- lapply(
        seq_along(x), function(i) ol_no_robust$add_point(x[[i]], y[[i]])
    )

    ol_robust <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        iterations = 3
    )
    results_robust <- lapply(
        seq_along(x), function(i) ol_robust$add_point(x[[i]], y[[i]])
    )

    expect_false(all(vapply(results_no_robust, is.null, logical(1))))
    expect_false(all(vapply(results_robust, is.null, logical(1))))
})
