# Shared helpers for the stored golden fixtures under tests/testthat/fixtures.
# `fixtures/make_reference.R` uses these to regenerate the committed CSVs;
# `test-golden.R` uses them to re-run each case and compare against the
# committed values. Every case reseeds the RNG and pins `parallel = FALSE` so
# the results are deterministic across machines and builds.

golden_seed <- function() {
    20240911L
}

# The single deterministic reference series shared by every case. Re-seeding
# here means a fresh run reproduces the exact values stored in the fixtures,
# independent of the caller's prior RNG state.
golden_series <- function(n = 60L) {
    set.seed(golden_seed())
    x <- as.double(seq(-3, 3, length.out = n))
    y <- as.double(sin(x) * x + rnorm(n, 0, 0.25))
    list(x = x, y = y)
}

# Write a data frame to CSV with 17 significant digits so numeric values
# survive a write/read round trip without loss.
golden_write_csv <- function(df, file) {
    cols <- lapply(seq_along(df), function(j) {
        v <- df[[j]]
        if (is.double(v)) sprintf("%.17g", v) else as.character(v)
    })
    out <- do.call(data.frame, c(cols, list(stringsAsFactors = FALSE)))
    names(out) <- names(df)
    utils::write.csv(out, file, row.names = FALSE, quote = FALSE, na = "")
}

golden_read_csv <- function(file) {
    utils::read.csv(file, stringsAsFactors = FALSE)
}

# One case per engine path that has no stats::lowess / stats::lm analogue:
# the default `boundary_policy = "extend"`, intervals, robustness weights,
# and the streaming and online engines.

golden_batch_default <- function(d) {
    model <- Lowess(
        fraction = 0.67,
        iterations = 3L,
        boundary_policy = "extend",
        parallel = FALSE
    )
    res <- fit(model, d$x, d$y)
    data.frame(x = d$x, y = d$y, yhat = res$y)
}

golden_batch_intervals <- function(d) {
    model <- Lowess(
        fraction = 0.3,
        iterations = 2L,
        boundary_policy = "extend",
        return_se = TRUE,
        confidence_intervals = 0.9,
        prediction_intervals = 0.9,
        return_derivative = TRUE,
        parallel = FALSE
    )
    res <- fit(model, d$x, d$y)
    data.frame(
        x = d$x,
        y = d$y,
        yhat = res$y,
        se = res$standard_errors,
        ci_lower = res$confidence_lower,
        ci_upper = res$confidence_upper,
        pi_lower = res$prediction_lower,
        pi_upper = res$prediction_upper,
        derivative = res$derivative
    )
}

golden_batch_robust <- function(d) {
    y <- d$y
    y[c(9L, 33L)] <- y[c(9L, 33L)] + 4
    model <- Lowess(
        fraction = 0.4,
        iterations = 5L,
        boundary_policy = "extend",
        return_robustness_weights = TRUE,
        parallel = FALSE
    )
    res <- fit(model, d$x, y)
    data.frame(
        x = d$x,
        y = y,
        yhat = res$y,
        robustness_weight = res$robustness_weights
    )
}

golden_streaming_chunked <- function(d) {
    sl <- StreamingLowess(
        fraction = 0.3,
        chunk_size = 20L,
        overlap = 0L,
        iterations = 1L,
        parallel = FALSE
    )
    res <- process_chunk(sl, d$x, d$y)
    fin <- finalize(sl)
    data.frame(
        x = c(as.double(res$x), as.double(fin$x)),
        y = d$y,
        yhat = c(as.double(res$y), as.double(fin$y))
    )
}

golden_online_full <- function(d) {
    ol <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 16L,
        min_points = 4L,
        update_mode = "full",
        iterations = 2L
    )
    out <- lapply(seq_along(d$x), function(i) add_point(ol, d$x[[i]], d$y[[i]]))
    yhat <- vapply(
        out,
        function(r) if (is.null(r)) NA_real_ else r$y,
        numeric(1)
    )
    data.frame(x = d$x, y = d$y, yhat = yhat)
}

# Fixture basename -> reference data frame, keyed by CSV filename.
golden_cases <- function() {
    d <- golden_series()
    list(
        batch_default = golden_batch_default(d),
        batch_intervals = golden_batch_intervals(d),
        batch_robust = golden_batch_robust(d),
        streaming_chunked = golden_streaming_chunked(d),
        online_full = golden_online_full(d)
    )
}
