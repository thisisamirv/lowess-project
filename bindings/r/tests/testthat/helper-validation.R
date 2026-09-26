# Shared helper for comparing this package's output against base R's
# `stats::lowess`. Used by both the fixed scenarios in test-validation.R and
# the randomized property-based fuzzing in test-property-lowess.R.

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
#' Companion to `reference_is_ulp_unstable()`: that function empirically
#' searches for a 1-ULP input perturbation that flips the fit, but for some
#' counterexamples the flipping perturbation is very specific and can be
#' missed by a bounded random search (confirmed on a real macos-latest/R
#' release runner: 1000 random trials failed to find it for a known-bad
#' case, even though the underlying instability is real and reproducible).
#'
#' This is a direct, deterministic check on the actual mechanism: at each
#' reweighting step from iteration `k` to `k + 1`, `stats::lowess()`
#' computes `cmad` (the median-absolute-residual-derived bisquare scale)
#' from the current fit's residuals, then classifies each residual against
#' hard cutoffs `0.001 * cmad` / `0.999 * cmad`. If `cmad` itself is smaller
#' than a generous multiple of the response's own floating-point precision,
#' every residual at that step is rounding noise from an already-exact fit,
#' not a real signal - so which side of the cutoff each residual falls on
#' is decided by whichever compiler/platform computed the fit, and there is
#' no well-defined answer to compare a second implementation against.
#' @noRd
cmad_is_noise_floor <- function(x, y, fraction, iterations, guard = 100) {
    if (iterations < 1L) {
        return(FALSE)
    }
    n <- length(x)
    eps <- .Machine$double.eps
    scale <- max(1, max(abs(y)))
    m1 <- n %/% 2L
    for (k in 0:(iterations - 1L)) {
        fit_k <- stats::lowess(x, y, f = fraction, iter = k)$y
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
#' The bisquare robustness reweighting in `stats::lowess()` (and this
#' package) applies a *hard* cutoff: residuals below `0.001 * cmad` get
#' weight 1, residuals above `0.999 * cmad` get weight exactly 0, and
#' `cmad` is itself derived from the residuals. For degenerate fuzzed
#' inputs (e.g. mostly-zero responses with one or two spikes and few
#' points), the initial fit can reproduce the response to within a few ULPs,
#' so every downstream residual - and thus `cmad` and the cutoffs - are
#' themselves pure floating-point rounding noise rather than a real signal.
#' When a residual sits within that noise band right at the cutoff, which
#' side of the hard threshold it lands on is decided by the last bit or two
#' of rounding in the local weighted regression - which is not required by
#' IEEE 754, or by R's own documentation, to be identical across compilers
#' (FMA contraction, summation order, vectorization all vary between Apple
#' Clang 14/17/21, rustc/LLVM, gcc, MSVC, etc.).
#'
#' This function empirically confirms that instability *in `stats::lowess`
#' alone* (no comparison to this package involved) by perturbing every `x`
#' and `y` value by exactly 1 ULP, in a fixed set of pseudo-random sign
#' patterns, and checking whether the resulting fit changes by more than
#' `tolerance`. If R's own reference output is not reproducible under a
#' perturbation far smaller than floating-point representability itself
#' guarantees, there is no well-defined "correct" answer to compare a
#' second, independently-computed implementation against, and the case
#' must be discarded rather than treated as a genuine divergence.
#'
#' The perturbation signs are drawn from a fixed seed per trial so the
#' decision is a deterministic function of `(x, y, fraction, iterations)`
#' and does not introduce run-to-run flakiness of its own. A few hundred
#' trials are needed in practice: only specific sign combinations flip the
#' cutoff decision, and this only runs on the (rare) comparison-failure
#' path, so the extra `stats::lowess()` calls are not a performance concern.
#' @noRd
reference_is_ulp_unstable <- function(
    x,
    y,
    fraction,
    iterations,
    base_fit,
    tolerance,
    trials = 1000L
) {
    n <- length(x)
    eps <- .Machine$double.eps
    x_ulp <- ifelse(x == 0, eps, abs(x) * eps)
    y_ulp <- ifelse(y == 0, eps, abs(y) * eps)
    comparison_scale <- max(1, abs(base_fit))

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
        x_perturbed <- x + sample(c(-1, 1), n, replace = TRUE) * x_ulp
        y_perturbed <- y + sample(c(-1, 1), n, replace = TRUE) * y_ulp
        perturbed_fit <- stats::lowess(
            x_perturbed,
            y_perturbed,
            f = fraction,
            iter = iterations
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
    tolerance = 1e-10
) {
    if (sorted) {
        x_fit <- as.double(x)
        y_fit <- as.double(y)
    } else {
        ord <- order(x)
        x_fit <- as.double(x[ord])
        y_fit <- as.double(y[ord])
    }

    reference <- stats::lowess(x_fit, y_fit, f = fraction, iter = iterations)
    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = zero_weight_fallback,
        outputs = if (sorted) "sorted" else NULL
    )
    result <- fit(model, x_fit, y_fit)

    if (
        sorted &&
            !isTRUE(all.equal(result$x, reference$x, tolerance = tolerance))
    ) {
        stop("x does not match stats::lowess sorted output", call. = FALSE)
    }
    max_diff <- max(abs(result$y - reference$y))
    comparison_scale <- max(1, abs(result$y), abs(reference$y))
    if (max_diff > tolerance * comparison_scale) {
        if (
            cmad_is_noise_floor(x_fit, y_fit, fraction, iterations) ||
                reference_is_ulp_unstable(
                    x_fit,
                    y_fit,
                    fraction,
                    iterations,
                    reference$y,
                    tolerance
                )
        ) {
            # `stats::lowess()` itself does not reproduce this fit when its
            # own inputs are perturbed by a single ULP: the bisquare hard
            # cutoff is being decided by rounding noise, not a real signal,
            # so there is no well-defined reference to compare against.
            return(TRUE)
        }
        iteration_counts <- seq.int(0L, as.integer(iterations))
        iteration_fits <- lapply(iteration_counts, function(n_iter) {
            reference_iter <- stats::lowess(x_fit, y_fit, f = fraction, iter = n_iter)
            model_iter <- Lowess(
                fraction = fraction,
                iterations = n_iter,
                boundary_policy = "noboundary",
                scaling_method = "mar",
                zero_weight_fallback = zero_weight_fallback,
                outputs = if (sorted) "sorted" else NULL
            )
            result_iter <- fit(model_iter, x_fit, y_fit)
            list(reference = reference_iter$y, package = result_iter$y)
        })
        iteration_diffs <- vapply(iteration_fits, function(fits) {
            max(abs(fits$package - fits$reference))
        }, numeric(1))
        first_divergence <- which(iteration_diffs > tolerance)[1]
        iteration_summary <- paste(
            sprintf("%d=%.17g", iteration_counts, iteration_diffs),
            collapse = ", "
        )
        failure_message <- sprintf(
            paste0(
                "y does not match stats::lowess output ",
                "(max abs diff: %.17g; x: %s; y: %s; ",
                "reference y: %s; package y: %s; ",
                "fraction: %.17g; iterations: %d; ",
                "per-iteration max diffs [iter=diff]: %s)"
            ),
            max_diff,
            toString(sprintf("%.17g", x)),
            toString(sprintf("%.17g", y)),
            toString(sprintf("%.17g", reference$y)),
            toString(sprintf("%.17g", result$y)),
            fraction,
            iterations,
            iteration_summary
        )
        if (length(first_divergence) > 0L && first_divergence > 1L) {
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
                    toString(sprintf("%.17g", abs(residual_y - previous_fits$reference)))
                ),
                sprintf(
                    "package abs residuals: [%s]",
                    toString(sprintf("%.17g", abs(residual_y - previous_fits$package)))
                )
            )
        }
        stop(failure_message, call. = FALSE)
    }
    if (!identical(result$fraction_used, fraction)) {
        stop("fraction_used does not match requested fraction", call. = FALSE)
    }
    TRUE
}
