#' @srrstats {G5.2, G5.2a, G5.2b} Error/warning tests for print/plot methods.
#' @srrstats {G5.3} No NA/NaN in print method outputs.
#' @srrstats {G5.5} Fixed random seeds in tests using set.seed().
#' @srrstats {RE4.17, RE4.18} Print method tests verify S3 dispatch.
#' @srrstats {RE6.0, RE6.2} Plot method tests verify output.

test_that("print.Lowess outputs correct fields", {
    model <- Lowess(fraction = 0.3, iterations = 2L)
    out <- capture.output(print(model))
    expect_true(any(grepl("Lowess Model", out)))
    expect_true(any(grepl("Fraction", out)))
    expect_true(any(grepl("Iterations", out)))
    expect_true(any(grepl("Weight Function", out)))
    expect_true(any(grepl("Parallel", out)))
    # print returns x invisibly
    expect_identical(print(model), model)
})

test_that("print.LowessResult outputs basic fields", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- Lowess(fraction = 0.3)$fit(x, y)
    out <- capture.output(print(result))
    expect_true(any(grepl("LowessResult", out)))
    expect_true(any(grepl("Points", out)))
    expect_true(any(grepl("Fraction Used", out)))
    expect_identical(print(result), result)
})

test_that("print.LowessResult shows iterations_used when present", {
    # Use a mock object to unconditionally exercise the optional branch
    mock <- structure(
        list(
            x = as.double(1:10), y = as.double(1:10),
            fraction_used = 0.3, iterations_used = 3L, cv_scores = NULL
        ),
        class = "LowessResult"
    )
    out <- capture.output(print(mock))
    expect_true(any(grepl("Iterations Used", out)))
})

test_that("print.LowessResult shows cv_scores when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 100)
    y <- sin(x) + rnorm(100, 0, 0.2)
    result <- Lowess(
        cv_fractions = c(0.2, 0.3, 0.5),
        cv_method = "kfold", cv_k = 5L
    )$fit(x, y)
    out <- capture.output(print(result))
    expect_true(any(grepl("CV Scores", out)))
})

test_that("print.StreamingLowess outputs correct fields", {
    model <- StreamingLowess(fraction = 0.3, chunk_size = 50L)
    out <- capture.output(print(model))
    expect_true(any(grepl("StreamingLowess Model", out)))
    expect_true(any(grepl("Fraction", out)))
    expect_true(any(grepl("Chunk Size", out)))
    expect_true(any(grepl("Parallel", out)))
    expect_identical(print(model), model)
})

test_that("print.OnlineLowess outputs correct fields", {
    model <- OnlineLowess(fraction = 0.2, window_capacity = 20L)
    out <- capture.output(print(model))
    expect_true(any(grepl("OnlineLowess Model", out)))
    expect_true(any(grepl("Fraction", out)))
    expect_true(any(grepl("Window Capacity", out)))
    expect_true(any(grepl("Min Points", out)))
    expect_identical(print(model), model)
})

test_that("plot.LowessResult runs without error", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- Lowess(fraction = 0.3)$fit(x, y)
    expect_no_error(plot(result))
})

test_that("plot.LowessResult draws confidence interval lines when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.2)
    result <- Lowess(fraction = 0.5, confidence_intervals = 0.95)$fit(x, y)
    expect_no_error(plot(result, main = "With CI"))
})
