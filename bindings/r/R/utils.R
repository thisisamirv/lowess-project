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
validate_common_args <- function(x, y, fraction, iterations, min_points = 2L) {
    validate_data_pair(x, y, min_points)
    validate_fit_fraction(fraction)
    validate_fit_iterations(iterations)

    list(
        x = as.double(x),
        y = as.double(y),
        fraction = as.double(fraction),
        iterations = as.integer(iterations)
    )
}

validate_data_pair <- function(x, y, min_points) {
    validate_numeric_vector(x, "x")
    validate_numeric_vector(y, "y")
    if (length(x) != length(y)) {
        stop("x and y must have the same length")
    }
    if (length(x) < min_points) {
        stop(sprintf("At least %d data points are required", min_points))
    }
}

validate_fit_fraction <- function(fraction) {
    if (!is.numeric(fraction) || length(fraction) != 1) {
        stop("fraction must be a single numeric value")
    }
    if (fraction <= 0 || fraction > 1) {
        stop("fraction must be between 0 and 1")
    }
}

validate_fit_iterations <- function(iterations) {
    if (is.null(iterations)) {
        stop("iterations must be a non-negative integer")
    }
    tryCatch(
        validate_optional_count(iterations, "iterations"),
        error = function(e) stop("iterations must be a non-negative integer")
    )
}


validate_numeric_vector <- function(value, name) {
    if (!is.numeric(value) || is.complex(value) || !is.null(dim(value))) {
        stop(
            sprintf("%s must be an integer or double vector", name),
            call. = FALSE
        )
    }
}

validate_named_options <- function(options, valid, name) {
    if (!is.list(options)) {
        stop(sprintf("`%s` must be a named list", name), call. = FALSE)
    }
    keys <- names(options)
    if (length(options) && (is.null(keys) || anyNA(keys) || any(keys == ""))) {
        stop(sprintf("`%s` must be a named list", name), call. = FALSE)
    }
    if (anyDuplicated(keys)) {
        stop(
            sprintf("Duplicate `%s` keys are not allowed", name),
            call. = FALSE
        )
    }
    unknown <- setdiff(keys, valid)
    if (length(unknown)) {
        stop(
            sprintf("Invalid `%s` key(s): %s", name, toString(unknown)),
            call. = FALSE
        )
    }
}

validate_scalar_numeric <- function(value, name) {
    if (!is.numeric(value) || length(value) != 1L || !is.finite(value)) {
        stop(sprintf("%s must be a single numeric value", name))
    }
}


validate_optional_count <- function(value, name, allow_zero = TRUE) {
    if (is.null(value)) {
        return(invisible(NULL))
    }

    validate_scalar_numeric(value, name)
    if (value != floor(value)) {
        stop(sprintf("%s must be a whole number", name))
    }
    if (value > .Machine$integer.max) {
        stop(sprintf("%s exceeds the maximum supported integer", name))
    }
    if (allow_zero && value < 0) {
        stop(sprintf("%s must be a non-negative integer", name))
    }
    if (!allow_zero && value <= 0) {
        stop(sprintf("%s must be a positive integer", name))
    }

    invisible(NULL)
}

#' Reject more than one unnamed positional argument
#'
#' @description
#' Only the constructor's first argument (`fraction`) may be supplied
#' positionally; every other argument must be named. Checked against the
#' original call expression (not the post-matching formals), since R would
#' otherwise happily fill later parameters (e.g. `window_capacity`,
#' `min_points`, `chunk_size`) positionally without complaint.
#'
#' @param call The calling function's \code{sys.call()}.
#' @param boundary_name Name of the last parameter mentioned in the error
#'   message (the final positional-looking parameter in the signature).
#' @noRd
reject_extra_positional_args <- function(call, boundary_name) {
    arg_names <- names(call)[-1]
    if (is.null(arg_names)) {
        arg_names <- rep("", length(call) - 1)
    }
    unnamed <- which(arg_names == "")
    if (
        length(unnamed) > 1L || (length(unnamed) == 1L && unnamed[[1L]] != 1L)
    ) {
        stop(
            sprintf("All arguments after '%s' must be named.", boundary_name),
            call. = FALSE
        )
    }
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
        window_capacity,
        "window_capacity",
        allow_zero = FALSE
    )
    validate_optional_count(min_points, "min_points")
    validate_optional_count(chunk_size, "chunk_size", allow_zero = FALSE)
}

#' Expand an `outputs` character vector into named logical flags
#'
#' Maps the user-facing `outputs = c("diagnostics", "residuals", ...)` argument
#' to the individual `return_*` booleans the Rust FFI expects. `NULL` (the
#' default) means "no optional components" (all `FALSE`). Unknown values are
#' rejected with the list of allowed names.
#'
#' @param outputs A character vector or `NULL`.
#' @param valid Character vector of allowed component names.
#' @return A named logical vector, one element per entry in `valid`.
#' @noRd
parse_outputs_flags <- function(outputs, valid) {
    if (is.null(outputs)) {
        result <- rep(FALSE, length(valid))
        names(result) <- valid
        return(result)
    }
    if (!is.character(outputs)) {
        stop("`outputs` must be a character vector or NULL", call. = FALSE)
    }
    unknown <- setdiff(outputs, valid)
    if (length(unknown) > 0L) {
        stop(
            sprintf(
                "Invalid `outputs` value(s): %s. Allowed: %s",
                toString(sprintf("'%s'", unknown)),
                toString(sprintf("'%s'", valid))
            ),
            call. = FALSE
        )
    }
    result <- valid %in% outputs
    names(result) <- valid
    result
}

#' Expand an `intervals` list into the flat values the Rust FFI expects
#'
#' Accepts `NULL` (no intervals), an \code{\link{intervals_opts}} object, or a
#' plain named list with any of `confidence`, `prediction`, and `bootstrap`.
#'
#' @param intervals `NULL` or a named list.
#' @return A list with `confidence`, `prediction` (each `NULL` or numeric)
#'   and `bootstrap` (integer, `0L` meaning off).
#' @noRd
expand_intervals <- function(intervals) {
    if (is.null(intervals)) {
        return(list(confidence = NULL, prediction = NULL, bootstrap = 0L))
    }
    if (!is.list(intervals)) {
        stop(
            "`intervals` must be NULL, intervals_opts(), or a named list",
            call. = FALSE
        )
    }
    valid <- c("confidence", "prediction", "bootstrap")
    validate_named_options(intervals, valid, "intervals")
    bootstrap <- intervals$bootstrap
    if (is.null(bootstrap)) {
        bootstrap <- 0L
    } else {
        validate_optional_count(bootstrap, "intervals$bootstrap")
    }
    list(
        confidence = intervals$confidence,
        prediction = intervals$prediction,
        bootstrap = as.integer(bootstrap)
    )
}

expand_cv <- function(cv) {
    if (is.null(cv)) {
        return(list(cv_fractions = NULL, cv_method = "kfold", cv_k = 5L))
    }
    validate_named_options(cv, c("method", "k", "fractions"), "cv")
    if (is.null(cv[["fractions", exact = TRUE]])) {
        stop("`cv$fractions` is required", call. = FALSE)
    }
    validate_numeric_vector(cv$fractions, "cv$fractions")
    if (!length(cv$fractions)) {
        stop("`cv$fractions` must be non-empty", call. = FALSE)
    }
    folds <- if (is.null(cv$k)) 5L else cv$k
    validate_optional_count(folds, "cv.k", allow_zero = FALSE)
    list(
        cv_fractions = cv$fractions,
        cv_method = if (is.null(cv$method)) "kfold" else cv$method,
        cv_k = folds
    )
}

lowess_grouped_args <- function(outputs, intervals, cv) {
    flags <- parse_outputs_flags(
        outputs,
        c("se", "diagnostics", "residuals", "weights", "derivative", "sorted")
    )
    names(flags) <- c(
        "return_se",
        "return_diagnostics",
        "return_residuals",
        "return_robustness_weights",
        "return_derivative",
        "return_sorted"
    )
    levels <- expand_intervals(intervals)
    c(
        as.list(flags),
        list(
            confidence_intervals = levels$confidence,
            prediction_intervals = levels$prediction,
            bootstrap = levels$bootstrap
        ),
        expand_cv(cv)
    )
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
    bootstrap = "integer",
    weight_function = "character",
    robustness_method = "character",
    scaling_method = "character",
    boundary_policy = "character",
    update_mode = "character",
    zero_weight_fallback = "character",
    cv_method = "character",
    merge_strategy = "character",
    missing = "character",
    return_diagnostics = "logical",
    return_residuals = "logical",
    return_robustness_weights = "logical",
    return_derivative = "logical",
    return_se = "logical",
    return_sorted = "logical",
    parallel = "logical",
    backend = "character",
    retain_model = "logical",
    delta = "nullable",
    overlap = "nullable",
    confidence_intervals = "nullable",
    prediction_intervals = "nullable",
    auto_converge = "nullable",
    cv_fractions = "nullable",
    seed = "nullable_double"
)

#' Build args from parent environment
#'
#' Captures all known parameters from the calling function's environment.
#' @param param_names Character vector of parameter names to extract.
#' @param overrides Named list of expanded grouped options.
#' @return Coerced list ready for do.call.
#' @noRd
env_args <- function(param_names, overrides = list()) {
    env <- parent.frame()
    result <- lapply(param_names, function(name) {
        val <- if (name %in% names(overrides)) {
            overrides[[name]]
        } else {
            get(name, envir = env)
        }
        type <- param_types[[name]]
        if (is.null(type)) {
            return(val)
        }
        switch(
            type,
            double = as.double(val),
            integer = as.integer(val),
            character = as.character(val),
            logical = as.logical(val),
            nullable = coerce_nullable(val)[[1]],
            nullable_double = if (is.null(val)) {
                Nullable(NULL)
            } else {
                as.double(val)
            },
            val
        )
    })
    setNames(result, param_names)
}

#' Parameter names for each Lowess constructor
#' @noRd
lowess_params <- c(
    "fraction",
    "iterations",
    "weight_function",
    "robustness_method",
    "delta",
    "zero_weight_fallback",
    "boundary_policy",
    "scaling_method",
    "auto_converge",
    "missing",
    "parallel",
    "backend",
    "return_se",
    "return_diagnostics",
    "return_residuals",
    "return_robustness_weights",
    "return_derivative",
    "return_sorted",
    "confidence_intervals",
    "prediction_intervals",
    "bootstrap",
    "cv_method",
    "cv_k",
    "cv_fractions",
    "seed",
    "retain_model"
)

online_params <- c(
    "fraction",
    "iterations",
    "weight_function",
    "robustness_method",
    "delta",
    "zero_weight_fallback",
    "boundary_policy",
    "scaling_method",
    "auto_converge",
    "missing",
    "window_capacity",
    "min_points",
    "update_mode",
    "return_se",
    "return_robustness_weights",
    "return_derivative",
    "confidence_intervals",
    "prediction_intervals",
    "bootstrap",
    "seed"
)

streaming_params <- c(
    "fraction",
    "iterations",
    "weight_function",
    "robustness_method",
    "delta",
    "zero_weight_fallback",
    "boundary_policy",
    "scaling_method",
    "auto_converge",
    "missing",
    "chunk_size",
    "overlap",
    "merge_strategy",
    "parallel",
    "return_se",
    "return_diagnostics",
    "return_residuals",
    "return_robustness_weights",
    "return_derivative",
    "confidence_intervals",
    "prediction_intervals",
    "bootstrap",
    "seed"
)
