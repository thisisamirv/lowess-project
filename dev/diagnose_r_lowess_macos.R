# Diagnostic script for the macos-latest / R-release-only property test
# failure in bindings/r/tests/testthat/test-property-lowess.R.
#
# Run from the repo root after installing the rfastlowess R package:
#   Rscript dev/diagnose_r_lowess_macos.R
#
# Prints: toolchain/compiler info, the two known failing counterexamples run
# through both stats::lowess() and this package (with full per-iteration
# diagnostics from helper-validation.R), and whether stats::lowess() itself
# is stable under 1-ULP input perturbation for those cases (see
# `reference_is_ulp_unstable()` for why that matters).

section <- function(title) {
    cat("\n", strrep("=", 70), "\n", title, "\n", strrep("=", 70), "\n", sep = "")
}

section("Toolchain / environment")
print(R.version)
cat("\n")
print(Sys.info())
cat("\nR CMD config CC:  ", system("R CMD config CC", intern = TRUE), "\n")
cat("R CMD config CXX: ", system("R CMD config CXX", intern = TRUE), "\n")
cc <- system("R CMD config CC", intern = TRUE)[1]
cc_bin <- strsplit(cc, "\\s+")[[1]][1]
cat("\n", cc_bin, " --version:\n", sep = "")
system(paste(cc_bin, "--version"))
cat("\nrustc --version:\n")
system("rustc --version")
cat("\nsessionInfo():\n")
print(sessionInfo())

section("Loading package + helpers")
suppressPackageStartupMessages(library(rfastlowess))
source("bindings/r/tests/testthat/helper-validation.R")
cat("rfastlowess loaded from:", find.package("rfastlowess"), "\n")

run_case <- function(label, x_raw, y_raw, fraction, iterations) {
    section(label)
    ord <- order(x_raw)
    x <- x_raw[ord]
    y <- y_raw[ord]

    reference <- stats::lowess(x, y, f = fraction, iter = iterations)
    model <- Lowess(
        fraction = fraction,
        iterations = as.integer(iterations),
        boundary_policy = "noboundary",
        scaling_method = "mar",
        zero_weight_fallback = "return_original"
    )
    result <- fit(model, x, y)

    max_diff <- max(abs(result$y - reference$y))
    cat(sprintf("max abs diff (reference vs package): %.17g\n", max_diff))
    cat("reference$y:", toString(sprintf("%.17g", reference$y)), "\n")
    cat("package$y:  ", toString(sprintf("%.17g", result$y)), "\n")

    unstable <- reference_is_ulp_unstable(
        x, y, fraction, iterations, reference$y, tolerance = 1e-5
    )
    cat(sprintf(
        "stats::lowess() itself unstable under 1-ULP input perturbation: %s\n",
        unstable
    ))

    ok <- tryCatch(
        {
            check_stats_lowess(
                x_raw, y_raw,
                fraction = fraction, iterations = iterations,
                zero_weight_fallback = "return_original", tolerance = 1e-5
            )
        },
        error = function(e) conditionMessage(e)
    )
    cat("check_stats_lowess() result:\n", paste(ok, collapse = "\n"), "\n")

    invisible(list(max_diff = max_diff, unstable = unstable))
}

run_case(
    "Case 1 (originally failed at iterations=10 on macos-latest/R release)",
    c(-1.2238545473664999, 0, 1.8667703326791525, 1.3440157659351826, 1.2536231782287359, 1.7515924870967865),
    c(0, 0, 0, 0, -0.022356696426868439, 1.2979496661573648),
    0.525, 10
)

run_case(
    "Case 2 (originally failed at iterations=2 on macos-latest/R release)",
    c(-1.3956062216311693, 0, -1.908623619005084, -0.94953064806759357, 1.6228566858917475, -0.34411230683326721),
    c(0, 0, 0, -0.774141451343894, 0, -1.637501435354352),
    0.525, 2
)

section("Full property test suite (with the reference-instability discard fix)")
results <- testthat::test_file(
    "bindings/r/tests/testthat/test-property-lowess.R",
    package = "rfastlowess",
    reporter = "summary"
)
print(results)
