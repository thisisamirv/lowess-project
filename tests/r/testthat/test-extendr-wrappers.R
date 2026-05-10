test_that("RLowess generated accessors dispatch fit methods", {
    null_value <- Nullable(NULL)
    handle <- rfastlowess:::RLowess$new(
        0.3, 1L, null_value, "tricube", "bisquare", "mad", "extend",
        null_value, null_value, FALSE, FALSE, FALSE, "use_local_mean",
        null_value, null_value, "kfold", 5L, FALSE
    )

    x <- as.double(1:10)
    y <- as.double(2 * x)
    result_dollar <- handle$fit(x, y)
    result_bracket <- handle[["fit"]](x, y)

    expect_equal(length(result_dollar$y), length(y))
    expect_equal(result_dollar$y, result_bracket$y)
})


test_that("RStreamingLowess generated accessors dispatch chunked methods", {
    null_value <- Nullable(NULL)
    handle <- rfastlowess:::RStreamingLowess$new(
        0.3, 10L, null_value, 1L, null_value, "tricube", "bisquare", "mad",
        "extend", null_value, FALSE, FALSE, FALSE
    )

    x <- as.double(1:10)
    y <- as.double(sin(x))
    chunk_result <- handle$process_chunk(x[1:5], y[1:5])
    final_result <- handle[["finalize"]]()

    expect_type(chunk_result, "list")
    expect_type(final_result, "list")
})


test_that("ROnlineLowess generated accessors dispatch add_points", {
    null_value <- Nullable(NULL)
    handle_dollar <- rfastlowess:::ROnlineLowess$new(
        0.3, 20L, 3L, 1L, null_value, "tricube", "bisquare", "mad",
        "extend", "incremental", null_value, FALSE, FALSE
    )
    handle_bracket <- rfastlowess:::ROnlineLowess$new(
        0.3, 20L, 3L, 1L, null_value, "tricube", "bisquare", "mad",
        "extend", "incremental", null_value, FALSE, FALSE
    )

    x <- as.double(1:10)
    y <- as.double(cos(x))
    result_dollar <- handle_dollar$add_points(x, y)
    result_bracket <- handle_bracket[["add_points"]](x, y)

    expect_equal(length(result_dollar$y), length(y))
    expect_equal(result_dollar$y, result_bracket$y)
})