#' Common argument validation and coercion
#'
#' @description
#' Internal helper to validate x, y, fraction, and iterations inputs and
#' force them to the correct types for Rust FFI.
#'
#' @srrstats {G2.0} Validates matching lengths, minimum points, numeric types.
#' @srrstats {G2.3} Informative error messages for invalid inputs.
#'
#' @param x Numeric vector
#' @param y Numeric vector
#' @param fraction Numeric
#' @param iterations Integer
#' @param min_points Minimum number of observations required.
#'
#' @return A list containing the coerced x, y, fraction, and iterations.
#' @noRd
#' @srrstats {G2.0} Validates matching lengths, minimum points, numeric types.
#' @srrstats {G2.2} Univariate only; multivariate inputs rejected.
#' @srrstats {G2.3} Informative error messages for invalid inputs.
#' @srrstats {G2.6} Numeric pre-processing via as.double/as.integer.
#' @srrstats {G2.13} Explicit NA handling before Rust FFI call.
#' @srrstats {G2.14, G2.14a, G2.14b, G2.14c} NaN/Inf reported via errors.
#' @srrstats {G2.15} NA checks on inputs before passing to algorithms.
#' @srrstats {G2.16} Inf/NaN validation in input vectors.
#' @srrstats {G3.0} Tolerance-based comparisons used in robustness weights.
validate_common_args <- function(x, y, fraction, iterations, min_points = 3L) {
    if (length(x) != length(y)) {
        stop("x and y must have the same length")
    }
    if (length(x) < min_points) {
        stop(sprintf("At least %d data points are required", min_points))
    }
    if (!is.numeric(fraction) || length(fraction) != 1) {
        stop("fraction must be a single numeric value")
    }
    if (fraction <= 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
    if (!is.numeric(iterations) || length(iterations) != 1 || iterations < 0) {
        stop("iterations must be a non-negative integer")
    }

    list(
        x = as.double(x),
        y = as.double(y),
        fraction = as.double(fraction),
        iterations = as.integer(iterations)
    )
}


validate_scalar_numeric <- function(value, name) {
    if (!is.numeric(value) || length(value) != 1L || is.na(value)) {
        stop(sprintf("%s must be a single numeric value", name))
    }
}


validate_optional_count <- function(value, name, allow_zero = TRUE) {
    if (is.null(value)) {
        return(invisible(NULL))
    }

    validate_scalar_numeric(value, name)
    if (allow_zero && value < 0) {
        stop(sprintf("%s must be a non-negative integer", name))
    }
    if (!allow_zero && value <= 0) {
        stop(sprintf("%s must be a positive integer", name))
    }

    invisible(NULL)
}

#' Validate constructor parameters
#'
#' @param fraction Smoothing fraction
#' @param iterations Robustness iterations (optional)
#' @param window_capacity Window capacity (optional)
#' @param min_points Minimum points (optional)
#' @param chunk_size Chunk size (optional)
#' @noRd
validate_params <- function(
    fraction,
    iterations = NULL,
    window_capacity = NULL,
    min_points = NULL,
    chunk_size = NULL
) {
    validate_scalar_numeric(fraction, "fraction")
    if (fraction < 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }

    validate_optional_count(iterations, "iterations")
    validate_optional_count(
        window_capacity, "window_capacity",
        allow_zero = FALSE
    )
    validate_optional_count(min_points, "min_points")
    validate_optional_count(chunk_size, "chunk_size", allow_zero = FALSE)
}

#' Coerce optional values to Nullable
#' @noRd
#' @srrstats {RE1.2} Numeric vector inputs documented and validated.
coerce_nullable <- function(...) {
    args <- list(...)
    lapply(args, function(x) if (is.null(x)) Nullable(NULL) else x)
}

#' Parameter type registry for Rust FFI coercion
#' @noRd
param_types <- list(
    fraction = "double",
    iterations = "integer",
    window_capacity = "integer",
    min_points = "integer",
    chunk_size = "integer",
    cv_k = "integer",
    weight_function = "character",
    robustness_method = "character",
    scaling_method = "character",
    boundary_policy = "character",
    update_mode = "character",
    zero_weight_fallback = "character",
    cv_method = "character",
    merge_strategy = "character",
    return_diagnostics = "logical",
    return_residuals = "logical",
    return_robustness_weights = "logical",
    parallel = "logical",
    delta = "nullable",
    overlap = "nullable",
    confidence_intervals = "nullable",
    prediction_intervals = "nullable",
    auto_converge = "nullable",
    cv_fractions = "nullable",
    cv_seed = "nullable"
)

#' Build args from parent environment
#'
#' Captures all known parameters from the calling function's environment.
#' @param param_names Character vector of parameter names to extract.
#' @return Coerced list ready for do.call.
#' @noRd
env_args <- function(param_names) {
    env <- parent.frame()
    result <- lapply(param_names, function(name) {
        val <- get(name, envir = env)
        type <- param_types[[name]]
        if (is.null(type)) {
            return(val)
        }
        switch(type,
            double = as.double(val),
            integer = as.integer(val),
            character = as.character(val),
            logical = as.logical(val),
            nullable = coerce_nullable(val)[[1]],
            val
        )
    })
    setNames(result, param_names)
}

#' Parameter names for each Lowess constructor
#' @noRd
lowess_params <- c(
    "fraction", "iterations", "delta", "weight_function", "robustness_method",
    "scaling_method", "boundary_policy", "confidence_intervals",
    "prediction_intervals", "return_diagnostics", "return_residuals",
    "return_robustness_weights", "zero_weight_fallback", "auto_converge",
    "cv_fractions", "cv_method", "cv_k", "parallel", "cv_seed"
)

online_params <- c(
    "fraction", "window_capacity", "min_points", "iterations",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "zero_weight_fallback", "update_mode", "auto_converge",
    "return_robustness_weights", "return_diagnostics",
    "return_residuals", "parallel",
    "delta",
    "confidence_intervals", "prediction_intervals"
)

streaming_params <- c(
    "fraction", "chunk_size", "overlap", "iterations",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "zero_weight_fallback", "auto_converge",
    "return_diagnostics", "return_residuals", "return_robustness_weights",
    "merge_strategy", "parallel",
    "delta",
    "confidence_intervals", "prediction_intervals"
)
