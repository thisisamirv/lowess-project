test_that("fastlowess_online() basic functionality works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    result <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25, min_points = 10
    )

    expect_type(result, "list")
    expect_equal(length(result$x), length(x))
    expect_equal(length(result$y), length(y))
    expect_type(result$x, "double")
    expect_type(result$y, "double")
})

test_that("fastlowess_online() window capacity works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    result_small <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 15
    )
    result_large <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 50
    )

    expect_equal(length(result_small$y), length(y))
    expect_equal(length(result_large$y), length(y))
})

test_that("fastlowess_online() min_points parameter works", {
    set.seed(42)
    x <- 1:50
    y <- sin(x / 10) + rnorm(50, sd = 0.1)

    result <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25, min_points = 5
    )

    expect_equal(length(result$y), 50)

    # First few points (before min_points) should be original values
    # (or close to them if smoothing starts immediately)
    expect_type(result$y, "double")
})

test_that("fastlowess_online() update modes work", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)

    result_full <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25,
        update_mode = "full"
    )
    result_incr <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25,
        update_mode = "incremental"
    )

    expect_equal(length(result_full$y), length(y))
    expect_equal(length(result_incr$y), length(y))
})

test_that("fastlowess_online() handles edge cases", {
    # Minimum data points
    x <- 1:10
    y <- 1:10
    result <- fastlowess_online(x, y,
        fraction = 0.5, window_capacity = 5, min_points = 3
    )
    expect_equal(length(result$y), 10)

    # Window larger than data
    result2 <- fastlowess_online(x, y,
        fraction = 0.5, window_capacity = 20, min_points = 3
    )
    expect_equal(length(result2$y), 10)
})

test_that("fastlowess_online() robustness works", {
    set.seed(42)
    x <- 1:100
    y <- sin(x / 10) + rnorm(100, sd = 0.1)
    y[50] <- y[50] + 5 # Add outlier

    result_no_robust <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25,
        iterations = 0
    )
    result_robust <- fastlowess_online(x, y,
        fraction = 0.3, window_capacity = 25,
        iterations = 3
    )

    expect_equal(length(result_no_robust$y), length(y))
    expect_equal(length(result_robust$y), length(y))
})
