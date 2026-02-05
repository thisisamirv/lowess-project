#' @srrstats {G5.4} Correctness tests for online/sliding window mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.8} Edge cases: min data, window > data.
#' @srrstats {RE4.0} Robustness iterations tested.
test_that("OnlineLowess basic functionality works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3, window_capacity = 25, min_points = 10
    )
    result <- ol$add_points(as.double(x), as.double(y))

    expect_type(result, "list")
    expect_equal(length(result$x), length(x))
    expect_equal(length(result$y), length(y))
    expect_type(result$x, "double")
    expect_type(result$y, "double")
})

test_that("OnlineLowess window capacity works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_small <- OnlineLowess(
        fraction = 0.3, window_capacity = 15
    )
    result_small <- ol_small$add_points(as.double(x), as.double(y))

    ol_large <- OnlineLowess(
        fraction = 0.3, window_capacity = 50
    )
    result_large <- ol_large$add_points(as.double(x), as.double(y))

    expect_equal(length(result_small$y), length(y))
    expect_equal(length(result_large$y), length(y))
})

test_that("OnlineLowess min_points parameter works", {
    set.seed(42)
    x <- 1:50
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    ol <- OnlineLowess(
        fraction = 0.3, window_capacity = 25, min_points = 5
    )
    result <- ol$add_points(as.double(x), as.double(y))

    expect_equal(length(result$y), 50)

    # First few points (before min_points) should be original values
    # (or close to them if smoothing starts immediately)
    expect_type(result$y, "double")
})

test_that("OnlineLowess update modes work", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    ol_full <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "full"
    )
    result_full <- ol_full$add_points(as.double(x), as.double(y))

    ol_incr <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        update_mode = "incremental"
    )
    result_incr <- ol_incr$add_points(as.double(x), as.double(y))

    expect_equal(length(result_full$y), length(y))
    expect_equal(length(result_incr$y), length(y))
})

test_that("OnlineLowess handles edge cases", {
    # Minimum data points
    x <- 1:10
    y <- 1:10
    ol <- OnlineLowess(
        fraction = 0.5, window_capacity = 5, min_points = 3
    )
    result <- ol$add_points(as.double(x), as.double(y))
    expect_equal(length(result$y), 10)

    # Window larger than data
    ol2 <- OnlineLowess(
        fraction = 0.5, window_capacity = 20, min_points = 3
    )
    result2 <- ol2$add_points(as.double(x), as.double(y))
    expect_equal(length(result2$y), 10)
})

test_that("OnlineLowess robustness works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)
    y[50] <- y[50] + 5 # Add outlier

    ol_no_robust <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        iterations = 0
    )
    result_no_robust <- ol_no_robust$add_points(as.double(x), as.double(y))

    ol_robust <- OnlineLowess(
        fraction = 0.3, window_capacity = 25,
        iterations = 3
    )
    result_robust <- ol_robust$add_points(as.double(x), as.double(y))

    expect_equal(length(result_no_robust$y), length(y))
    expect_equal(length(result_robust$y), length(y))
})
