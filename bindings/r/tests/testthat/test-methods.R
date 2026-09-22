#' @srrstats {G5.2, G5.2a, G5.2b} Error/warning tests for print/plot methods.
#' @srrstats {G5.3} No NA/NaN in print method outputs.
#' @srrstats {G5.5} Fixed random seeds in tests using set.seed().
#' @srrstats {RE4.17, RE4.18} Print method tests verify S3 dispatch.
#' @srrstats {RE6.0, RE6.2} Plot method tests verify output.

test_that("print.Lowess outputs correct fields", {
    model <- Lowess(fraction = 0.3, iterations = 2L)
    out <- capture.output(print(model))
    expect_true(any(grepl("Lowess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Iterations", out, fixed = TRUE)))
    expect_true(any(grepl("Weight Function", out, fixed = TRUE)))
    expect_true(any(grepl("Parallel", out, fixed = TRUE)))
    # print returns x invisibly
    expect_identical(print(model), model)
})

test_that("print.LowessResult outputs basic fields", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- fit(Lowess(fraction = 0.3), x, y)
    out <- capture.output(print(result))
    expect_true(any(grepl("LowessResult", out, fixed = TRUE)))
    expect_true(any(grepl("Points", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction Used", out, fixed = TRUE)))
    expect_identical(print(result), result)
})

test_that("print.LowessResult shows iterations_used when present", {
    # Use a mock object to unconditionally exercise the optional branch
    mock <- structure(
        list(
            x = as.double(1:10),
            y = as.double(1:10),
            fraction_used = 0.3,
            iterations_used = 3L,
            cv_scores = NULL
        ),
        class = "LowessResult"
    )
    out <- capture.output(print(mock))
    expect_true(any(grepl("Iterations Used", out, fixed = TRUE)))
})

test_that("print.LowessResult shows cv_scores when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 100)
    y <- sin(x) + rnorm(100, 0, 0.2)
    result <- fit(
        Lowess(
            cv = cv_opts(fractions = c(0.2, 0.3, 0.5), method = "kfold", k = 5L)
        ),
        x,
        y
    )
    out <- capture.output(print(result))
    expect_true(any(grepl("CV Scores", out, fixed = TRUE)))
})

test_that("print.StreamingLowess outputs correct fields", {
    model <- StreamingLowess(fraction = 0.3, chunk_size = 50L)
    out <- capture.output(print(model))
    expect_true(any(grepl("StreamingLowess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Chunk Size", out, fixed = TRUE)))
    expect_true(any(grepl("Parallel", out, fixed = TRUE)))
    expect_identical(print(model), model)
})

test_that("print.OnlineLowess outputs correct fields", {
    model <- OnlineLowess(fraction = 0.2, window_capacity = 20L)
    out <- capture.output(print(model))
    expect_true(any(grepl("OnlineLowess Model", out, fixed = TRUE)))
    expect_true(any(grepl("Fraction", out, fixed = TRUE)))
    expect_true(any(grepl("Window Capacity", out, fixed = TRUE)))
    expect_true(any(grepl("Min Points", out, fixed = TRUE)))
    expect_identical(print(model), model)
})

test_that("plot.LowessResult runs without error", {
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    result <- fit(Lowess(fraction = 0.3), x, y)
    expect_no_error(plot(result))
})

test_that("plot.LowessResult draws confidence interval lines when present", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.2)
    result <- fit(Lowess(fraction = 0.5, confidence_intervals = 0.95), x, y)
    expect_no_error(plot(result, main = "With CI"))
})

test_that("fit.Lowess errors on unused ... arguments", {
    model <- Lowess(fraction = 0.3)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x)
    expect_error(fit(model, x, y, extra = 1), "unused arguments")
})

test_that("process_chunk.StreamingLowess errors on unused ... arguments", {
    model <- StreamingLowess(fraction = 0.3, chunk_size = 50L)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x)
    expect_error(process_chunk(model, x, y, extra = 1), "unused arguments")
})

test_that("finalize.StreamingLowess errors on unused ... arguments", {
    model <- StreamingLowess(fraction = 0.3, chunk_size = 50L)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x)
    invisible(process_chunk(model, x, y))
    expect_error(finalize(model, extra = 1), "unused arguments")
})

test_that("add_point.OnlineLowess errors on unused ... arguments", {
    model <- OnlineLowess(fraction = 0.3, window_capacity = 20L)
    expect_error(add_point(model, 1.0, 0.5, extra = 1), "unused arguments")
})

# ── predict.Lowess ────────────────────────────────────────────────────────────

test_that("predict.Lowess returns predicted y values", {
    set.seed(42)
    x <- seq(0, 10, length.out = 100)
    y <- sin(x) + rnorm(100, 0, 0.1)
    model <- Lowess(fraction = 0.2, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(2.5, 7.5))

    expect_type(pred, "list")
    expect_named(pred, "y")
    expect_length(pred$y, 2)
    expect_true(all(is.finite(pred$y)))
})

test_that("predict.Lowess matches fit() at training points", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    result <- fit(model, x, y)

    pred <- predict(model, x)

    expect_length(pred$y, length(x))
    expect_equal(pred$y, result$y, tolerance = 1e-6)
})

test_that("predict.Lowess return_se returns standard errors", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(2.5, 7.5), outputs = "se")

    expect_true("standard_errors" %in% names(pred))
    expect_length(pred$standard_errors, 2)
    expect_true(all(is.finite(pred$standard_errors)))
    expect_true(all(pred$standard_errors >= 0))
})

test_that("predict.Lowess confidence_level returns confidence intervals", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(2.5, 7.5), confidence_level = 0.95)

    expect_true("confidence_lower" %in% names(pred))
    expect_true("confidence_upper" %in% names(pred))
    expect_length(pred$confidence_lower, 2)
    expect_length(pred$confidence_upper, 2)
    # Confidence bounds bracket the fitted values
    expect_lte(max(pred$confidence_lower - pred$y), 0)
    expect_gte(min(pred$confidence_upper - pred$y), 0)
})

test_that("predict.Lowess prediction_level returns prediction intervals", {
    set.seed(42)
    x <- seq(0, 10, length.out = 50)
    y <- sin(x) + rnorm(50, 0, 0.1)
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(2.5, 7.5), prediction_level = 0.95)

    expect_true("prediction_lower" %in% names(pred))
    expect_true("prediction_upper" %in% names(pred))
    expect_length(pred$prediction_lower, 2)
    expect_length(pred$prediction_upper, 2)
})

test_that("predict.Lowess return_derivative returns local slopes", {
    x <- as.double(0:59)
    y <- 3.0 * x - 1.0
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(20, 30, 40), outputs = "derivative")

    expect_true("derivative" %in% names(pred))
    expect_length(pred$derivative, 3)
    expect_equal(pred$derivative, rep(3.0, 3), tolerance = 1e-6)
})

test_that("predict.Lowess clamps out-of-range queries by default", {
    x <- as.double(0:29)
    y <- 0.5 * x
    model <- Lowess(fraction = 0.4, retain_model = TRUE)
    invisible(fit(model, x, y))

    pred <- predict(model, c(-100, -1, 1000))

    expect_length(pred$y, 3)
    expect_true(all(is.finite(pred$y)))
})

test_that("extrapolation = 'error' rejects out-of-range queries", {
    x <- as.double(0:29)
    y <- 0.5 * x
    model <- Lowess(fraction = 0.4, retain_model = TRUE)
    invisible(fit(model, x, y))

    expect_error(
        predict(model, 1000, extrapolation = "error"),
        "outside the training range"
    )
    # In-range queries still succeed under the same policy
    expect_no_error(predict(model, 5, extrapolation = "error"))
})

test_that("linear extrapolation respects max_extrapolation_distance", {
    x <- as.double(0:49)
    y <- 2.0 * x + 3.0
    model <- Lowess(
        fraction = 0.3,
        boundary_policy = "noboundary",
        retain_model = TRUE
    )
    invisible(fit(model, x, y))

    expect_no_error(
        predict(
            model,
            55,
            extrapolation = "linear",
            max_extrapolation_distance = 10
        )
    )
    expect_error(
        predict(
            model,
            100,
            extrapolation = "linear",
            max_extrapolation_distance = 10
        ),
        "max_extrapolation_distance"
    )
})

test_that("predict.Lowess max_neighbor_distance catches sparse neighborhoods", {
    x <- as.double(c(0:10, 90:100))
    y <- 2.0 * x
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, x, y))

    # Uncapped prediction in a gap silently succeeds
    expect_no_error(predict(model, 50))
    # A cap catches the gap
    expect_error(
        predict(model, 50, max_neighbor_distance = 5),
        "max_neighbor_distance"
    )
    # A point near real training data still succeeds under the cap
    expect_no_error(predict(model, 5, max_neighbor_distance = 5))
})

test_that("predict.Lowess rejects non-finite new_x", {
    x <- as.double(0:29)
    y <- 0.5 * x
    model <- Lowess(fraction = 0.4, retain_model = TRUE)
    invisible(fit(model, x, y))

    expect_error(predict(model, c(5, NaN)), "new_x")
})

test_that("predict.Lowess errors on unused ... arguments", {
    model <- Lowess(fraction = 0.3, retain_model = TRUE)
    invisible(fit(model, as.double(1:10), as.double(1:10)))

    expect_error(predict(model, 5, bogus = 1), "unused arguments")
})

test_that("predict.Lowess errors without retain_model = TRUE", {
    model <- Lowess(fraction = 0.3, retain_model = FALSE)
    invisible(fit(model, as.double(1:10), as.double(1:10)))

    expect_error(predict(model, 5), "retain_model")
})
