# Shared helper for comparing this package's output against base R's
# `stats::lowess`. Used by both the fixed scenarios in test-validation.R and
# the randomized property-based fuzzing in test-property-lowess.R.

lowess_reference_counts <- new.env(parent = emptyenv())
lowess_reference_counts$compared <- 0L
lowess_reference_counts$too_short <- 0L
lowess_reference_counts$noise_floor <- 0L
lowess_reference_counts$ulp_unstable <- 0L
lowess_reference_counts$failures <- list()

#' Assert that this package reproduces `stats::lowess` on a fixed dataset.
#'
#' Pins R's comparison settings: no boundary padding (`"noboundary"`),
#' MAR residual scaling (`"mar"`), and R's zero-weight behavior
#' (`zero_weight_fallback = "return_original"`). The package default remains
#' `"use_local_mean"`; comparisons must opt into R's fallback explicitly.
#' `x` is sorted ascending first, as `stats::lowess` assumes.
#'
#' @param direct If `TRUE`, request an exact (non-interpolated) surface by
#'   setting `delta = 0` on both sides; otherwise each side uses its default
#'   (1% of the x-range).
#' @param zero_weight_fallback Fallback used for the `stats::lowess`
#'   comparison. Must stay `"return_original"` to match R.
#' @param tolerance Relative tolerance; observed agreement is ~1e-15.
#' @noRd
expect_matches_stats_lowess <- function(
    x,
    y,
    fraction,
    iterations,
    direct = FALSE,
    zero_weight_fallback = "return_original",
    tolerance = 1e-10
) {
    ord <- order(x)
    x <- as.double(x[ord])
    y <- as.double(y[ord])

    delta <- if (direct) 0.0 else NULL

    reference <- if (is.null(delta)) {
        stats::lowess(x, y, f = fraction, iter = iterations)
    } else {
        stats::lowess(x, y, f = fraction, iter = iterations, delta = delta)
    }

    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        delta = delta,
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = zero_weight_fallback
    )
    result <- fit(model, x, y)

    expect_equal(result$y, reference$y, tolerance = tolerance)
    expect_identical(result$fraction_used, fraction)

    invisible(result)
}

#' Assert that `outputs = "sorted"` reproduces `stats::lowess` output order.
#'
#' Unlike `expect_matches_stats_lowess()`, this intentionally passes unsorted
#' inputs through to `Lowess()` and checks both `x` and `y` against
#' `stats::lowess()`'s always-sorted return value.
#' @noRd
expect_stats_lowess_sorted <- function(
    x,
    y,
    fraction,
    iterations,
    direct = FALSE,
    zero_weight_fallback = "return_original",
    tolerance = 1e-10
) {
    x <- as.double(x)
    y <- as.double(y)

    delta <- if (direct) 0.0 else NULL

    reference <- if (is.null(delta)) {
        stats::lowess(x, y, f = fraction, iter = iterations)
    } else {
        stats::lowess(x, y, f = fraction, iter = iterations, delta = delta)
    }

    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        delta = delta,
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = zero_weight_fallback,
        outputs = "sorted"
    )
    result <- fit(model, x, y)

    expect_equal(result$x, reference$x, tolerance = tolerance)
    expect_equal(result$y, reference$y, tolerance = tolerance)
    expect_identical(result$fraction_used, fraction)

    invisible(result)
}

#' Check whether `stats::lowess()`'s own bisquare scale is noise, not signal.
#'
#' Deterministic companion to `reference_is_ulp_unstable()`, whose bounded
#' random search can miss the specific perturbation that flips a fit. If `cmad`
#' at any reweighting step falls below the response's own floating-point
#' precision, the residuals there are rounding noise, so which side of the
#' `0.001`/`0.999 * cmad` cutoffs each lands on is decided by the compiler and
#' there is no well-defined answer to compare against.
#' @noRd
cmad_is_noise_floor <- function(
    x,
    y,
    fraction,
    iterations,
    guard = 100,
    delta = NULL
) {
    if (iterations < 1L) {
        return(FALSE)
    }
    ord <- order(x)
    x <- as.double(x[ord])
    y <- as.double(y[ord])
    resolved_delta <- if (is.null(delta)) 0.01 * diff(range(x)) else delta
    n <- length(x)
    eps <- .Machine$double.eps
    scale <- max(1, max(abs(y)))
    m1 <- n %/% 2L
    for (k in 0:(iterations - 1L)) {
        fit_k <- stats::lowess(
            x,
            y,
            f = fraction,
            iter = k,
            delta = resolved_delta
        )$y
        s <- sort(abs(y - fit_k))
        cmad <- if (n %% 2L == 0L) {
            m2 <- n - m1 - 1L
            3 * (s[m1 + 1L] + s[m2 + 1L])
        } else {
            6 * s[m1 + 1L]
        }
        if (cmad < guard * eps * scale) {
            return(TRUE)
        }
    }
    FALSE
}

#' Check whether `stats::lowess()` itself is stable under 1-ULP input noise.
#'
#' Bisquare reweighting applies hard cutoffs at `0.001`/`0.999 * cmad`, so a
#' residual sitting in the rounding-noise band can fall either side depending on
#' compiler details (FMA contraction, summation order, vectorization). This
#' perturbs R's own inputs by 1 ULP, with no comparison to this package: if R
#' cannot reproduce its own fit, there is no well-defined reference and the case
#' must be discarded rather than treated as a divergence. Perturbation signs use
#' a fixed seed, so the verdict introduces no run-to-run flakiness. It runs only
#' on the rare failure path, so the extra fits do not matter for performance.
#' @noRd
reference_is_ulp_unstable <- function(
    x,
    y,
    fraction,
    iterations,
    base_fit,
    tolerance,
    trials = 1000L,
    delta = NULL
) {
    ord <- order(x)
    x <- as.double(x[ord])
    y <- as.double(y[ord])
    resolved_delta <- if (is.null(delta)) 0.01 * diff(range(x)) else delta
    n <- length(x)
    eps <- .Machine$double.eps
    x_ulp <- ifelse(x == 0, eps, abs(x) * eps)
    y_ulp <- ifelse(y == 0, eps, abs(y) * eps)
    comparison_scale <- max(1, abs(base_fit))

    # `stats::lowess` branches on exact equality of x, so splitting tied values
    # apart would probe a different class of input.
    # Perturb each level as a unit.
    x_level <- match(x, unique(x))
    n_levels <- max(x_level)

    has_seed <- exists(".Random.seed", envir = .GlobalEnv)
    old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
        if (has_seed) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (exists(".Random.seed", envir = .GlobalEnv)) {
            rm(".Random.seed", envir = .GlobalEnv)
        }
    })
    set.seed(0L)

    for (trial in seq_len(trials)) {
        x_signs <- sample(c(-1, 1), n_levels, replace = TRUE)[x_level]
        x_perturbed <- x + x_signs * x_ulp
        y_perturbed <- y + sample(c(-1, 1), n, replace = TRUE) * y_ulp
        perturbed_fit <- stats::lowess(
            x_perturbed,
            y_perturbed,
            f = fraction,
            iter = iterations,
            delta = resolved_delta
        )$y
        if (max(abs(perturbed_fit - base_fit)) > tolerance * comparison_scale) {
            return(TRUE)
        }
    }
    FALSE
}

#' Check `Lowess()` against `stats::lowess()` without testthat expectations.
#'
#' `quickcheck` treats signalled `testthat` success conditions as falsifying in
#' some shrink paths, so property tests use this base-R checker instead.
#' @noRd
check_stats_lowess <- function(
    x,
    y,
    fraction,
    iterations,
    sorted = FALSE,
    zero_weight_fallback = "return_original",
    tolerance = 1e-10,
    delta = NULL,
    parallel = TRUE
) {
    x_fit <- as.double(x)
    y_fit <- as.double(y)
    if (length(x_fit) < 2L) {
        lowess_reference_counts$too_short <- lowess_reference_counts$too_short +
            1L
        hedgehog::discard()
    }
    ord <- order(x_fit)
    resolved_delta <- if (is.null(delta)) 0.01 * diff(range(x_fit)) else delta

    reference <- stats::lowess(
        x_fit,
        y_fit,
        f = fraction,
        iter = iterations,
        delta = resolved_delta
    )
    expected_x <- if (sorted) reference$x else x_fit
    expected_y <- if (sorted) reference$y else reference$y[order(ord)]
    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        delta = resolved_delta,
        parallel = parallel,
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = zero_weight_fallback,
        outputs = if (sorted) "sorted" else NULL
    )
    result <- fit(model, x_fit, y_fit)

    if (!isTRUE(all.equal(result$x, expected_x, tolerance = 0))) {
        stop(
            "x does not match the requested LOWESS output order",
            call. = FALSE
        )
    }
    max_diff <- max(abs(result$y - expected_y))
    comparison_scale <- max(1, abs(result$y), abs(expected_y))
    if (max_diff > tolerance * comparison_scale) {
        if (
            cmad_is_noise_floor(
                x_fit,
                y_fit,
                fraction,
                iterations,
                delta = resolved_delta
            )
        ) {
            lowess_reference_counts$noise_floor <- lowess_reference_counts$noise_floor +
                1L
            hedgehog::discard()
        }
        if (
            reference_is_ulp_unstable(
                x_fit,
                y_fit,
                fraction,
                iterations,
                reference$y,
                tolerance,
                delta = resolved_delta
            )
        ) {
            lowess_reference_counts$ulp_unstable <- lowess_reference_counts$ulp_unstable +
                1L
            hedgehog::discard()
        }
        lowess_reference_counts$failures[[
            length(lowess_reference_counts$failures) + 1L
        ]] <- list(
            x = x_fit,
            y = y_fit,
            fraction = fraction,
            iterations = iterations,
            sorted = sorted,
            zero_weight_fallback = zero_weight_fallback,
            tolerance = tolerance,
            delta = resolved_delta,
            parallel = parallel
        )
        iteration_counts <- seq.int(0L, as.integer(iterations))
        iteration_fits <- lapply(iteration_counts, function(n_iter) {
            reference_iter <- stats::lowess(
                x_fit,
                y_fit,
                f = fraction,
                iter = n_iter,
                delta = resolved_delta
            )
            model_iter <- Lowess(
                fraction = fraction,
                iterations = n_iter,
                delta = resolved_delta,
                parallel = parallel,
                boundary_policy = "noboundary",
                scaling_method = "mar",
                zero_weight_fallback = zero_weight_fallback,
                outputs = if (sorted) "sorted" else NULL
            )
            result_iter <- fit(model_iter, x_fit, y_fit)
            expected_iter <- if (sorted) {
                reference_iter$y
            } else {
                reference_iter$y[order(ord)]
            }
            list(reference = expected_iter, package = result_iter$y)
        })
        iteration_diffs <- vapply(
            iteration_fits,
            function(fits) {
                max(abs(fits$package - fits$reference))
            },
            numeric(1)
        )
        first_divergence <- which(
            iteration_diffs > tolerance * comparison_scale
        )[1]
        iteration_summary <- toString(
            sprintf("%d=%.17g", iteration_counts, iteration_diffs)
        )
        failure_message <- sprintf(
            paste0(
                "y does not match stats::lowess output ",
                "(max abs diff: %.17g; x: %s; y: %s; ",
                "reference y: %s; package y: %s; ",
                "fraction: %.17g; iterations: %d; delta: %.17g; parallel: %s; ",
                "per-iteration max diffs [iter=diff]: %s)"
            ),
            max_diff,
            toString(sprintf("%.17g", x)),
            toString(sprintf("%.17g", y)),
            toString(sprintf("%.17g", expected_y)),
            toString(sprintf("%.17g", result$y)),
            fraction,
            iterations,
            resolved_delta,
            parallel,
            iteration_summary
        )
        if (!is.na(first_divergence) && first_divergence > 1L) {
            previous_fits <- iteration_fits[[first_divergence - 1L]]
            residual_y <- if (sorted) y_fit[order(x_fit)] else y_fit
            failure_message <- paste0(
                failure_message,
                sprintf(
                    "; last agreeing iteration %d reference fit: [%s]; ",
                    iteration_counts[first_divergence - 1L],
                    toString(sprintf("%.17g", previous_fits$reference))
                ),
                sprintf(
                    "package fit: [%s]; reference abs residuals: [%s]; ",
                    toString(sprintf("%.17g", previous_fits$package)),
                    toString(sprintf(
                        "%.17g",
                        abs(residual_y - previous_fits$reference)
                    ))
                ),
                sprintf(
                    "package abs residuals: [%s]",
                    toString(sprintf(
                        "%.17g",
                        abs(residual_y - previous_fits$package)
                    ))
                )
            )
        }
        stop(failure_message, call. = FALSE)
    }
    if (!identical(result$fraction_used, fraction)) {
        stop("fraction_used does not match requested fraction", call. = FALSE)
    }
    lowess_reference_counts$compared <- lowess_reference_counts$compared + 1L
    TRUE
}
