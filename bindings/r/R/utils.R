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
#'
#' @return A list containing the coerced x, y, fraction, and iterations.
#' @noRd
validate_common_args <- function(x, y, fraction, iterations) {
    if (length(x) != length(y)) {
        stop("x and y must have the same length")
    }
    if (length(x) < 3) {
        stop("At least 3 data points are required")
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
    if (fraction < 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
    if (!is.null(iterations) && iterations < 0) {
        stop("iterations must be a non-negative integer")
    }
    if (!is.null(window_capacity) && window_capacity <= 0) {
        stop("window_capacity must be a positive integer")
    }
    if (!is.null(min_points) && min_points < 0) {
        stop("min_points must be a non-negative integer")
    }
    if (!is.null(chunk_size) && chunk_size <= 0) {
        stop("chunk_size must be a positive integer")
    }
}

#' Coerce optional values to Nullable
#' @noRd
coerce_nullable <- function(...) {
    args <- list(...)
    lapply(args, function(x) if (is.null(x)) Nullable(NULL) else x)
}

#' Parameter type registry for Rust FFI coercion
#' @noRd
param_types <- list(
    # Numeric parameters
    fraction = "double",
    # Integer parameters
    iterations = "integer",
    window_capacity = "integer",
    min_points = "integer",
    chunk_size = "integer",
    cv_k = "integer",
    # Character parameters
    weight_function = "character",
    robustness_method = "character",
    scaling_method = "character",
    boundary_policy = "character",
    update_mode = "character",
    zero_weight_fallback = "character",
    cv_method = "character",
    # Logical parameters
    return_diagnostics = "logical",
    return_residuals = "logical",
    return_robustness_weights = "logical",
    parallel = "logical",
    # Nullable parameters (optional, NULL -> Nullable(NULL))
    delta = "nullable",
    overlap = "nullable",
    confidence_intervals = "nullable",
    prediction_intervals = "nullable",
    auto_converge = "nullable",
    cv_fractions = "nullable"
)

#' Coerce named parameters for Rust FFI
#'
#' Uses non-standard evaluation to capture variable names.
#' Pass variables directly: coerce_params(fraction, iterations, delta)
#' @param ... Variables to coerce (names are captured automatically).
#' @return A list of coerced values in the order passed.
#' @noRd
coerce_params <- function(...) {
    # Capture the unevaluated call to get variable names
    exprs <- as.list(substitute(list(...)))[-1]
    names <- vapply(exprs, deparse, character(1))
    # Evaluate to get values
    args <- list(...)
    lapply(seq_along(args), function(i) {
        name <- names[i]
        val <- args[[i]]
        type <- param_types[[name]]
        if (is.null(type)) {
            return(val)
        } # Unknown param, pass through
        switch(type,
            double = as.double(val),
            integer = as.integer(val),
            character = as.character(val),
            logical = as.logical(val),
            nullable = if (is.null(val)) Nullable(NULL) else val,
            val
        )
    })
}

#' Build args from parent environment
#'
#' Captures all known parameters from the calling function's environment.
#' @param param_names Character vector of parameter names to extract.
#' @return Coerced list ready for do.call.
#' @noRd
env_args <- function(param_names) {
    env <- parent.frame()
    lapply(param_names, function(name) {
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
            nullable = if (is.null(val)) Nullable(NULL) else val,
            val
        )
    })
}

#' Parameter names for each Lowess constructor
#' @noRd
lowess_params <- c(
    "fraction", "iterations", "delta", "weight_function", "robustness_method",
    "scaling_method", "boundary_policy", "confidence_intervals",
    "prediction_intervals", "return_diagnostics", "return_residuals",
    "return_robustness_weights", "zero_weight_fallback", "auto_converge",
    "cv_fractions", "cv_method", "cv_k", "parallel"
)

online_params <- c(
    "fraction", "window_capacity", "min_points", "iterations", "delta",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "update_mode", "auto_converge",
    "return_robustness_weights", "parallel"
)

streaming_params <- c(
    "fraction", "chunk_size", "overlap", "iterations", "delta",
    "weight_function", "robustness_method", "scaling_method",
    "boundary_policy", "auto_converge", "return_diagnostics",
    "return_robustness_weights", "parallel"
)
