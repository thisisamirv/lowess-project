RLowess <- getFromNamespace("RLowess", "rfastlowess")
RStreamingLowess <- getFromNamespace("RStreamingLowess", "rfastlowess")
ROnlineLowess <- getFromNamespace("ROnlineLowess", "rfastlowess")

test_that("RLowess generated accessors dispatch fit methods", {
    null_value <- Nullable(NULL)
    handle <- RLowess$new(
        fraction = 0.3, iterations = 1L, delta = null_value,
        weight_function = "tricube", robustness_method = "bisquare",
        scaling_method = "mad", boundary_policy = "extend",
        confidence_intervals = null_value, prediction_intervals = null_value,
        return_diagnostics = FALSE, return_residuals = FALSE,
        return_robustness_weights = FALSE,
        zero_weight_fallback = "use_local_mean", auto_converge = null_value,
        cv_fractions = null_value, cv_method = "kfold", cv_k = 5L,
        parallel = FALSE, cv_seed = null_value, return_se = FALSE
    )

    x <- as.double(1:10)
    y <- as.double(2 * x)
    result_dollar <- handle$fit(x, y)
    result_bracket <- handle[["fit"]](x, y)

    expect_length(result_dollar$y, length(y))
    expect_identical(result_dollar$y, result_bracket$y)
})


test_that("RStreamingLowess generated accessors dispatch chunked methods", {
    null_value <- Nullable(NULL)
    handle <- RStreamingLowess$new(
        fraction = 0.3, chunk_size = 10L, overlap = null_value,
        iterations = 1L,
        weight_function = "tricube", robustness_method = "bisquare",
        scaling_method = "mad", boundary_policy = "extend",
        zero_weight_fallback = "use_local_mean", auto_converge = null_value,
        return_diagnostics = FALSE, return_residuals = FALSE,
        return_robustness_weights = FALSE,
        merge_strategy = "weighted_average", parallel = FALSE,
        delta = null_value, confidence_intervals = null_value,
        prediction_intervals = null_value
    )

    x <- as.double(1:10)
    y <- as.double(sin(x))
    chunk_result <- handle$process_chunk(x[1:5], y[1:5])
    final_result <- handle[["finalize"]]()

    expect_type(chunk_result, "list")
    expect_type(final_result, "list")
})


test_that("ROnlineLowess generated accessors dispatch add_point", {
    null_value <- Nullable(NULL)
    handle_dollar <- ROnlineLowess$new(
        fraction = 0.3, window_capacity = 20L, min_points = 3L,
        iterations = 1L,
        weight_function = "tricube", robustness_method = "bisquare",
        scaling_method = "mad", boundary_policy = "extend",
        zero_weight_fallback = "use_local_mean", update_mode = "incremental",
        auto_converge = null_value, return_robustness_weights = FALSE,
        return_diagnostics = FALSE, return_residuals = FALSE,
        parallel = FALSE, delta = null_value,
        confidence_intervals = null_value, prediction_intervals = null_value
    )
    handle_bracket <- ROnlineLowess$new(
        fraction = 0.3, window_capacity = 20L, min_points = 3L,
        iterations = 1L,
        weight_function = "tricube", robustness_method = "bisquare",
        scaling_method = "mad", boundary_policy = "extend",
        zero_weight_fallback = "use_local_mean", update_mode = "incremental",
        auto_converge = null_value, return_robustness_weights = FALSE,
        return_diagnostics = FALSE, return_residuals = FALSE,
        parallel = FALSE, delta = null_value,
        confidence_intervals = null_value, prediction_intervals = null_value
    )

    # Prime both handles with enough points to get a result
    for (i in 1:10) {
        handle_dollar$add_point(as.double(i), cos(as.double(i)))
        handle_bracket[["add_point"]](as.double(i), cos(as.double(i)))
    }

    result_dollar <- handle_dollar$add_point(11.0, cos(11.0))
    result_bracket <- handle_bracket[["add_point"]](11.0, cos(11.0))

    # Both should return the same type (list or NULL)
    expect_identical(is.null(result_dollar), is.null(result_bracket))
    if (!is.null(result_dollar)) {
        expect_type(result_dollar, "list")
        expect_type(result_bracket, "list")
        expect_true("smoothed" %in% names(result_dollar))
        expect_true("smoothed" %in% names(result_bracket))
    }
})
