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
cv_opts <- getFromNamespace("cv_opts", "rfastlowess")
parse_outputs_flags <- getFromNamespace("parse_outputs_flags", "rfastlowess")

# ── cv_opts ──────────────────────────────────────────────────────────────────

test_that("cv_opts validates required and non-empty fractions", {
    expect_error(
        cv_opts(),
        "`fractions` must be a numeric vector of candidate fractions"
    )
    expect_error(
        cv_opts(fractions = NULL),
        "`fractions` must be a numeric vector of candidate fractions"
    )
    expect_error(
        cv_opts(fractions = "0.5"),
        "`fractions` must be a non-empty numeric vector"
    )
    expect_error(
        cv_opts(fractions = numeric()),
        "`fractions` must be a non-empty numeric vector"
    )
})

test_that("cv_opts returns coerced cross-validation options", {
    result <- cv_opts(
        method = factor("kfold"),
        k = 3L,
        fractions = c(0.2, 0.5)
    )

    expect_s3_class(result, "cv_opts")
    expect_type(result$fractions, "double")
    expect_identical(result$fractions, c(0.2, 0.5))
    expect_identical(result$method, "kfold")
    expect_identical(result$k, 3L)
    expect_null(result$seed)
})

test_that("cv_opts rejects fractional fold counts", {
    expect_error(cv_opts(k = 3.8, fractions = c(0.2, 0.5)), "whole number")
})

test_that("intervals_opts and expand_intervals build grouped options", {
    result <- intervals_opts(confidence = 0.9, bootstrap = 20)
    expect_s3_class(result, "intervals_opts")
    expect_identical(result$bootstrap, 20L)
    expect_error(intervals_opts(bootstrap = 2.5), "whole number")

    expect_identical(
        expand_intervals(NULL),
        list(confidence = NULL, prediction = NULL, bootstrap = 0L)
    )
    expect_identical(
        expand_intervals(list(prediction = 0.8)),
        list(confidence = NULL, prediction = 0.8, bootstrap = 0L)
    )
    expect_error(expand_intervals(list(0.95)), "must be a named list")
    expect_error(expand_intervals(list(foo = 1)), "Invalid `intervals` key")
})

test_that("common validation rejects fractional iteration counts", {
    expect_error(
        validate_common_args(1:5, 1:5, fraction = 0.5, iterations = 1.5),
        "iterations must be a non-negative integer"
    )
})

test_that("all constructors validate iteration counts before coercion", {
    for (count in c(1.5, -0.5, Inf, NA_real_, .Machine$integer.max + 1)) {
        expect_error(Lowess(iterations = count), "iterations")
        expect_error(StreamingLowess(iterations = count), "iterations")
        expect_error(
            OnlineLowess(iterations = count, update_mode = "full"),
            "iterations"
        )
    }
    expect_error(
        Lowess(cv = list(k = 2.5, fractions = c(0.3, 0.5))),
        "whole number"
    )
})

test_that("integer custom weights match double custom weights", {
    model <- Lowess(fraction = 0.5, iterations = 0L)
    integer_result <- fit(model, 1:10, sin(1:10), custom_weights = rep(1L, 10))
    double_result <- fit(model, 1:10, sin(1:10), custom_weights = rep(1, 10))
    expect_equal(integer_result$y, double_result$y)
    expect_error(fit(model, 1:10, sin(1:10), custom_weights = -1L))
})

test_that("constructors reject named options in dots", {
    expect_error(Lowess(output = "se"), "unused arguments")
    expect_error(StreamingLowess(unknown = TRUE), "unused arguments")
    expect_error(OnlineLowess(output = "se"), "unused arguments")
})

test_that("numeric inputs are validated before coercion", {
    model <- Lowess(retain_model = TRUE)
    expect_error(fit(model, factor(1:10), sin(1:10)), "x must be")
    expect_error(fit(model, 1:10, factor(1:10)), "y must be")
    expect_error(fit(model, matrix(1:10, nrow = 2), sin(1:10)), "x must be")
    expect_error(
        fit(model, 1:10, sin(1:10), custom_weights = factor(rep("-1", 10))),
        "custom_weights must be"
    )
    expect_error(predict(model, factor(1:2)), "new_x must be")
    expect_error(add_point(OnlineLowess(), factor(1), 2), "x must be")
})

test_that("grouped options reject missing, unknown and duplicate keys", {
    expect_error(Lowess(cv = list(fraction = c(0.2, 0.5))), "Invalid `cv` key")
    expect_error(Lowess(cv = list(k = 3)), "fractions.*required")
    expect_error(Lowess(cv = list(fractions = numeric())), "non-empty")
    expect_error(cv_opts(fractions = matrix(c(0.2, 0.5))), "fractions must be")
    expect_error(
        Lowess(cv = list(fractions = c(0.2, 0.5), k = 2, k = 3)),
        "Duplicate `cv` keys"
    )
    expect_error(
        expand_intervals(list(confidence = 0.8, confidence = 0.99)),
        "Duplicate `intervals` keys"
    )
})

test_that("parse_outputs_flags handles NULL and valid output names", {
    valid <- c("diagnostics", "residuals", "weights")

    expect_identical(
        parse_outputs_flags(NULL, valid),
        setNames(c(FALSE, FALSE, FALSE), valid)
    )
    expect_identical(
        parse_outputs_flags(c("weights", "diagnostics"), valid),
        setNames(c(TRUE, FALSE, TRUE), valid)
    )
})

test_that("parse_outputs_flags rejects invalid output values", {
    valid <- c("diagnostics", "residuals", "weights")

    expect_error(
        parse_outputs_flags(1L, valid),
        "must be a character vector or NULL"
    )
    expect_error(
        parse_outputs_flags("unknown", valid),
        "Invalid `outputs` value"
    )
})

# ── validate_common_args ────────────────────────────────────────────────────

test_that("validate_common_args rejects mismatched lengths", {
    expect_error(
        validate_common_args(1:3, 1:4, 0.5, 3),
        "x and y must have the same length"
    )
})

test_that("validate_common_args rejects fewer than 2 points", {
    expect_error(
        validate_common_args(1, 1, 0.5, 3),
        "At least 2 data points are required"
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
        intervals = intervals_opts(confidence = 0.95),
        seed = 1
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
        update_mode = "incremental"
    )
    expect_s3_class(model, "OnlineLowess")
})
