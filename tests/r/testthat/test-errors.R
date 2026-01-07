test_that("fastlowess() rejects invalid inputs", {
    x <- 1:10
    y <- 1:10

    # Mismatched lengths
    expect_error(
        fastlowess(1:10, 1:5),
        "x and y must have the same length"
    )

    # Invalid fraction
    expect_error(
        fastlowess(x, y, fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        fastlowess(x, y, fraction = 1.5),
        "fraction must be between 0 and 1"
    )

    # Invalid iterations
    expect_error(
        fastlowess(x, y, iterations = -1),
        "iterations must be a non-negative integer"
    )
})

test_that("fastlowess_online() rejects invalid inputs", {
    x <- 1:10
    y <- 1:10

    # Mismatched lengths
    expect_error(
        fastlowess_online(1:10, 1:5),
        "x and y must have the same length"
    )

    # Invalid parameters
    expect_error(
        fastlowess_online(x, y, fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        fastlowess_online(x, y, window_capacity = 0),
        "window_capacity must be a positive integer"
    )
    expect_error(
        fastlowess_online(x, y, min_points = -1),
        "min_points must be a non-negative integer"
    )
})

test_that("fastlowess_streaming() rejects invalid inputs", {
    x <- 1:10
    y <- 1:10

    # Mismatched lengths
    expect_error(
        fastlowess_streaming(1:10, 1:5),
        "x and y must have the same length"
    )

    # Invalid parameters
    expect_error(
        fastlowess_streaming(x, y, fraction = -0.1),
        "fraction must be between 0 and 1"
    )
    expect_error(
        fastlowess_streaming(x, y, chunk_size = 0),
        "chunk_size must be a positive integer"
    )
})
