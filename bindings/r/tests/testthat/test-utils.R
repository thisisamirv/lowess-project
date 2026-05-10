#' @srrstats {G2.0, G2.2, G2.3} Validation tests for length, type, range.
#' @srrstats {G2.4} Type coercion verified in constructor tests.
#' @srrstats {G5.3} No NA/NaN in validated outputs.
#' @srrstats {G5.8, G5.8a, G5.8b, G5.8c, G5.8d} Edge condition tests.
# Tests targeting uncovered lines in utils.R:
#   validate_common_args (lines 18-39)
#   coerce_nullable (lines 77-78)
#   env_args: unknown-param passthrough (line 157)

validate_common_args <- getFromNamespace("validate_common_args", "rfastlowess")
validate_params <- getFromNamespace("validate_params", "rfastlowess")
coerce_nullable <- getFromNamespace("coerce_nullable", "rfastlowess")
env_args <- getFromNamespace("env_args", "rfastlowess")

# ── validate_common_args ────────────────────────────────────────────────────

test_that("validate_common_args rejects mismatched lengths", {
    expect_error(
        validate_common_args(1:3, 1:4, 0.5, 3),
        "x and y must have the same length"
    )
})

test_that("validate_common_args rejects fewer than 3 points", {
    expect_error(
        validate_common_args(1:2, 1:2, 0.5, 3),
        "At least 3 data points are required"
    )
})

test_that("validate_common_args rejects non-numeric fraction", {
    expect_error(
        validate_common_args(1:5, 1:5, "a", 3),
        "fraction must be a single numeric value"
    )
})

test_that("validate_common_args rejects fraction out of range", {
    expect_error(
        validate_common_args(1:5, 1:5, 0, 3),
        "fraction must be between 0 and 1"
    )
    expect_error(
        validate_common_args(1:5, 1:5, 1.5, 3),
        "fraction must be between 0 and 1"
    )
})

test_that("validate_common_args rejects negative iterations", {
    expect_error(
        validate_common_args(1:5, 1:5, 0.5, -1),
        "iterations must be a non-negative integer"
    )
})

test_that("validate_common_args returns coerced list on valid input", {
    result <- validate_common_args(1:5, 2:6, 0.5, 3)
    expect_type(result$x, "double")
    expect_type(result$y, "double")
    expect_type(result$fraction, "double")
    expect_type(result$iterations, "integer")
})

test_that("validate_params rejects NA scalar inputs", {
    expect_error(
        validate_params(NA_real_),
        "fraction must be a single numeric value"
    )
    expect_error(
        validate_params(0.5, iterations = NA_real_),
        "iterations must be a single numeric value"
    )
})

# ── coerce_nullable ─────────────────────────────────────────────────────────

test_that("coerce_nullable wraps NULL values", {
    result <- coerce_nullable(NULL, NULL)
    expect_null(result[[1]])
    expect_null(result[[2]])
})

test_that("coerce_nullable passes through non-NULL values unchanged", {
    result <- coerce_nullable(0.95, NULL)
    expect_identical(result[[1]], 0.95)
    expect_null(result[[2]])
})

# ── env_args: unknown-param passthrough (line 157) ──────────────────────────
# env_args returns val as-is when the param name is not in param_types.

test_that("env_args passes through unknown parameter names unchanged", {
    result <- local({
        my_unknown_param <- 42
        env_args("my_unknown_param")
    })
    expect_identical(result[[1]], 42)
})

test_that("env_args handles unknown types in param_types registry", {
    ns <- asNamespace("rfastlowess")
    orig_types <- ns$param_types

    # Temporarily inject a dummy type
    new_types <- orig_types
    new_types[["dummy_type_param"]] <- "unhandled_switch_type"

    # assignInNamespace handles unlocking/relocking internally for namespaces
    utils::assignInNamespace("param_types", new_types, "rfastlowess")
    on.exit(
        utils::assignInNamespace("param_types", orig_types, "rfastlowess"),
        add = TRUE
    )

    result <- local({
        dummy_type_param <- "test_value"
        env_args("dummy_type_param")
    })

    expect_identical(result[[1]], "test_value")
})

# ── constructor-level coverage of env_args type branches ────────────────────

test_that("Lowess constructor coerces all param types via env_args", {
    # Exercises double, integer, character, logical, nullable
    model <- Lowess(
        fraction = 0.4,
        iterations = 2L,
        weight_function = "tricube",
        parallel = FALSE,
        delta = NULL,
        confidence_intervals = 0.95
    )
    expect_s3_class(model, "Lowess")
    expect_identical(model$params$fraction, 0.4)
    expect_identical(model$params$iterations, 2L)
})

test_that("StreamingLowess constructor coerces overlap via env_args", {
    model <- StreamingLowess(fraction = 0.3, chunk_size = 50L, overlap = NULL)
    expect_s3_class(model, "StreamingLowess")
})

test_that("OnlineLowess constructor coerces all param types via env_args", {
    model <- OnlineLowess(
        fraction = 0.2,
        window_capacity = 20L,
        min_points = 3L,
        update_mode = "incremental",
        parallel = FALSE
    )
    expect_s3_class(model, "OnlineLowess")
})
