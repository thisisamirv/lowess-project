#' @srrstats {G5.4, G5.4b} Reference comparison against base R's
#'   `stats::lowess`, generalized here to randomized inputs.
#' @srrstats {G5.10} Property-based tests run in the standard suite.
#'   `quickcheck` is installed by `make validate`, not a package dependency.
#' @noRd

# Property-based regression test: fuzzes x/y/fraction/iterations and checks
# that this package's output matches `stats::lowess` (via
# `expect_matches_stats_lowess()` in helper-validation.R), pinning R's
# comparison settings (`boundary_policy = "noboundary"`,
# `scaling_method = "mar"`, `zero_weight_fallback = "return_original"`).
# The package default remains `"use_local_mean"`; these comparisons opt into
# R's fallback explicitly.
#
# This complements the fixed scenarios in test-validation.R: randomized
# comparisons can expose divergences at iteration counts not covered by any
# single hand-picked case.

# quickcheck shrinks below the declared minimum length, so the two-point
# minimum that both `stats::lowess` and this package require is enforced here.
usable_x <- function(x, min_length = 2L) {
    if (length(x) < min_length) {
        lowess_reference_counts$too_short <- lowess_reference_counts$too_short +
            1L
        hedgehog::discard()
    }
    TRUE
}

test_that("cmad noise-floor guard handles unsorted x", {
    x <- c(
        18.696686858311296,
        5.791397113353014,
        -61.549769470468163,
        71.320148417726159
    )
    y <- c(-11.987563783981185, 0, 0, 0)
    fraction <- 0.964674779062625
    iterations <- 168L
    ord <- order(x)

    expect_true(cmad_is_noise_floor(x[ord], y[ord], fraction, iterations))
    expect_identical(
        cmad_is_noise_floor(x, y, fraction, iterations),
        cmad_is_noise_floor(x[ord], y[ord], fraction, iterations)
    )
})

test_that("matches stats::lowess for randomized inputs (property-based)", {
    property <- function(xy, fraction, iterations) {
        usable_x(xy[[1]])
        expect_true(check_stats_lowess(
            xy[[1]],
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches stats::lowess for randomized sorted output", {
    property <- function(xy, fraction, iterations) {
        usable_x(xy[[1]])
        expect_true(check_stats_lowess(
            xy[[1]],
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 25L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

# Continuous draws practically never collide, so ties need their own generator.
# `stats::lowess` copies one fitted value across a run of tied x-values instead
# of refitting each point, and collapsing the draw onto a handful of levels
# exercises that path (including the all-tied case, where every x is equal).
test_that("matches stats::lowess for tied x-values (property-based)", {
    property <- function(xy, levels, fraction, iterations) {
        usable_x(xy[[1]])
        x <- round(xy[[1]] / (200 / levels))
        expect_true(check_stats_lowess(
            x,
            xy[[2]],
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(2L, 40L))
        ),
        levels = quickcheck::integer_bounded(1L, 8L, len = 1L),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches initial stats::lowess fits for sparse one-spike responses", {
    property <- function(
        x,
        spike_position,
        spike_magnitude,
        spike_negative,
        fraction,
        iterations
    ) {
        usable_x(x)

        spike_index <- min(
            length(x),
            floor(spike_position * length(x)) + 1L
        )
        spike_value <- if (spike_negative) -spike_magnitude else spike_magnitude
        y <- numeric(length(x))
        y[spike_index] <- spike_value

        expect_true(check_stats_lowess(
            x,
            y,
            fraction = fraction,
            iterations = iterations,
            sorted = TRUE,
            zero_weight_fallback = "return_original",
            tolerance = 1e-10
        ))
    }

    quickcheck::for_all(
        x = quickcheck::double_bounded(-100, 100, len = c(2L, 40L)),
        spike_position = quickcheck::double_bounded(0, 1, len = 1L),
        spike_magnitude = quickcheck::double_bounded(1e-4, 100, len = 1L),
        spike_negative = quickcheck::logical_(len = 1L),
        fraction = quickcheck::double_bounded(0.05, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 300L, len = 1L),
        property = property,
        tests = 200L,
        discards = 1000L
    )
})

test_that("matches stats::lowess at explicit delta and predictor-gap boundaries", {
    property <- function(samples, fraction, iterations, mode, parallel) {
        predictors <- samples[[1]]
        gaps <- diff(sort(unique(predictors)))
        gap <- if (length(gaps)) min(gaps) else 0
        span <- diff(range(predictors))
        delta <- switch(
            mode,
            0,
            NULL,
            0.01 * span,
            0.5 * span,
            gap * (1 - 1e-8),
            gap,
            gap * (1 + 1e-8)
        )

        expect_true(check_stats_lowess(
            predictors,
            samples[[2]],
            fraction,
            iterations,
            delta = delta,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-10, 10),
            quickcheck::double_bounded(-5, 5),
            len = c(12L, 40L)
        ),
        fraction = quickcheck::double_bounded(0.15, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
        mode = quickcheck::integer_bounded(1L, 7L, len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::lowess mapped back to original input order", {
    property <- function(samples, fraction, iterations, ordering, parallel) {
        indices <- switch(
            ordering,
            rev(seq_along(samples[[1]])),
            order(samples[[2]]),
            order(samples[[1]], decreasing = TRUE)
        )
        expect_true(check_stats_lowess(
            samples[[1]][indices],
            samples[[2]][indices],
            fraction,
            iterations,
            sorted = FALSE,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-10, 10),
            len = c(12L, 40L)
        ),
        fraction = quickcheck::double_bounded(0.2, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
        ordering = quickcheck::integer_bounded(1L, 3L, len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("serial and parallel LOWESS both match stats::lowess", {
    property <- function(samples, fraction, iterations, sorted, direct) {
        for (parallel in c(FALSE, TRUE)) {
            expect_true(check_stats_lowess(
                samples[[1]],
                samples[[2]],
                fraction,
                iterations,
                sorted = sorted,
                delta = if (direct) 0 else NULL,
                parallel = parallel,
                tolerance = 1e-10
            ))
        }
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100),
            quickcheck::double_bounded(-5, 5),
            len = c(16L, 48L)
        ),
        fraction = quickcheck::double_bounded(0.2, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
        sorted = quickcheck::logical_(len = 1L),
        direct = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::lowess across clusters, gaps and near ties", {
    property <- function(
        samples,
        layout,
        fraction,
        iterations,
        direct,
        parallel
    ) {
        coordinates <- samples[[1]]
        predictors <- switch(
            layout,
            coordinates / 100 + ifelse(coordinates < 0, -10, 10),
            round(coordinates),
            round(coordinates) + coordinates * 1e-10,
            round(coordinates, 1)
        )
        expect_true(check_stats_lowess(
            predictors,
            samples[[2]],
            fraction,
            iterations,
            delta = if (direct) 0 else NULL,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-5, 5),
            len = c(12L, 40L)
        ),
        layout = quickcheck::integer_bounded(1L, 4L, len = 1L),
        fraction = quickcheck::double_bounded(0.15, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 12L, len = 1L),
        direct = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::lowess at neighborhood and numerical-scale boundaries", {
    property <- function(
        samples,
        neighbors,
        direction,
        exponent,
        offset,
        response_exponent,
        direct,
        parallel
    ) {
        predictors <- (samples[[1]] + offset) * 10^exponent
        responses <- samples[[2]] * 10^response_exponent
        fraction <- (neighbors + direction * 1e-8) / length(predictors)

        expect_true(check_stats_lowess(
            predictors,
            responses,
            fraction,
            iterations = 0L,
            delta = if (direct) 0 else NULL,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-5, 5),
            len = c(16L, 48L)
        ),
        neighbors = quickcheck::integer_bounded(2L, 12L, len = 1L),
        direction = quickcheck::integer_bounded(-1L, 1L, len = 1L),
        exponent = quickcheck::integer_bounded(-8L, 8L, len = 1L),
        offset = quickcheck::double_bounded(-1e6, 1e6, len = 1L),
        response_exponent = quickcheck::integer_bounded(-4L, 8L, len = 1L),
        direct = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("matches stats::lowess for structured signals and multiple outliers", {
    property <- function(
        samples,
        shape,
        fraction,
        iterations,
        direct,
        parallel,
        odd
    ) {
        count <- length(samples[[1]])
        if (count %% 2L != as.integer(odd)) {
            count <- count - 1L
        }
        predictors <- samples[[1]][seq_len(count)]
        coordinates <- predictors / max(1, max(abs(predictors)))
        responses <- switch(
            shape,
            rep(2.5, count),
            1.5 * coordinates - 0.75,
            ifelse(coordinates < 0, -1, 1),
            rep(c(-1, 1), length.out = count),
            samples[[2]][seq_len(count)]
        )
        if (shape == 5L) {
            responses[c(1L, count)] <- responses[c(1L, count)] + c(20, -20)
        }
        expect_true(check_stats_lowess(
            predictors,
            responses,
            fraction,
            iterations,
            delta = if (direct) 0 else NULL,
            parallel = parallel,
            tolerance = 1e-10
        ))
    }
    quickcheck::for_all(
        samples = quickcheck::equal_length(
            quickcheck::double_bounded(-5, 5),
            quickcheck::double_bounded(-2, 2),
            len = c(16L, 48L)
        ),
        shape = quickcheck::integer_bounded(1L, 5L, len = 1L),
        fraction = quickcheck::double_bounded(0.2, 1.0, len = 1L),
        iterations = quickcheck::integer_bounded(0L, 25L, len = 1L),
        direct = quickcheck::logical_(len = 1L),
        parallel = quickcheck::logical_(len = 1L),
        odd = quickcheck::logical_(len = 1L),
        property = property,
        tests = 100L,
        shrinks = 0L,
        discards = 1000L
    )
})

test_that("counts LOWESS reference comparisons and discards explicitly", {
    expect_gt(lowess_reference_counts$compared, 0L)
    message(
        "LOWESS comparisons: ",
        lowess_reference_counts$compared,
        "; short-input discards: ",
        lowess_reference_counts$too_short,
        "; noise-floor discards: ",
        lowess_reference_counts$noise_floor,
        "; ULP-unstable discards: ",
        lowess_reference_counts$ulp_unstable,
        "; captured failures: ",
        length(lowess_reference_counts$failures)
    )
})
