#' @srrstats {G5.4} Correctness tests for streaming/chunked mode.
#' @srrstats {G5.5} Fixed random seeds.
#' @srrstats {G5.7} Parallel vs serial performance comparison.
#' @srrstats {G5.8} Edge cases: small data, chunk > data.
#' @srrstats {RE3.0} Diagnostics in streaming mode tested.
# Helper to simulate bulk streaming
bulk_stream <- function(x, y, ...) {
    sl <- StreamingLowess(...)
    res <- sl$process_chunk(as.double(x), as.double(y))
    fin <- sl$finalize()
    # Merge results
    list(
        x = c(res$x, fin$x),
        y = c(res$y, fin$y),
        diagnostics = if (!is.null(fin$diagnostics)) fin$diagnostics else NULL
    )
}

test_that("StreamingLowess basic functionality works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 1000)
    y <- sin(x) + rnorm(1000, sd = 0.1)

    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, overlap = 20
    )

    expect_type(result, "list")
    expect_equal(length(result$x), length(x))
    expect_equal(length(result$y), length(y))
})

test_that("StreamingLowess handles different chunk sizes", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- sin(x) + rnorm(500, sd = 0.1)

    result_small <- bulk_stream(x, y, fraction = 0.3, chunk_size = 100)
    result_large <- bulk_stream(x, y, fraction = 0.3, chunk_size = 250)

    expect_equal(length(result_small$y), length(y))
    expect_equal(length(result_large$y), length(y))

    # Results should be similar
    expect_equal(result_small$y, result_large$y, tolerance = 0.1)
})

test_that("StreamingLowess overlap parameter works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- sin(x) + rnorm(500, sd = 0.1)

    result_no_overlap <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100, overlap = 0
    )
    result_overlap <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100, overlap = 20
    )

    expect_equal(length(result_no_overlap$y), length(y))
    expect_equal(length(result_overlap$y), length(y))
})

test_that("StreamingLowess diagnostics work", {
    set.seed(42)
    x <- seq(0, 10, length.out = 500)
    y <- 2 * x + rnorm(500, sd = 0.5)

    result <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 100,
        return_diagnostics = TRUE
    )

    expect_true("diagnostics" %in% names(result))
    expect_type(result$diagnostics, "list")
})

test_that("StreamingLowess handles edge cases", {
    # Small dataset
    x <- 1:50
    y <- sin(x / 10)
    result <- bulk_stream(x, y, fraction = 0.3, chunk_size = 20)
    expect_equal(length(result$y), 50)

    # Chunk size larger than data
    result2 <- bulk_stream(x, y, fraction = 0.3, chunk_size = 100)
    expect_equal(length(result2$y), 50)
})

test_that("StreamingLowess parallel execution works", {
    set.seed(42)
    x <- seq(0, 10, length.out = 1000)
    y <- sin(x) + rnorm(1000, sd = 0.1)

    result_serial <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, parallel = FALSE
    )
    result_parallel <- bulk_stream(x, y,
        fraction = 0.3, chunk_size = 200, parallel = TRUE
    )

    # Results should be nearly identical
    expect_equal(result_serial$y, result_parallel$y, tolerance = 1e-8)
})
