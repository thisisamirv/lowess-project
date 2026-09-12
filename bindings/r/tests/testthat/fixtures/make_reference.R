# Regenerate the stored golden reference fixtures in this directory.
#
# These fixtures freeze the package's output for engine paths that have no
# external reference implementation (default `boundary_policy = "extend"`,
# intervals, robustness weights, streaming, online). They are committed so
# that any change - intended or not - shows up as a reviewed diff in `git`.
#
# Usage (run from the package root):
#   Rscript tests/testthat/fixtures/make_reference.R
#
# Requires a built, installed copy of rfastlowess. The script locates the
# fixtures directory from its own path, so the working directory does not
# matter.

args <- commandArgs(trailingOnly = FALSE)
self <- sub("^--file=", "", args[grep("^--file=", args)])
fixture_dir <- normalizePath(dirname(self))
testthat_dir <- normalizePath(file.path(fixture_dir, ".."))

if (!requireNamespace("rfastlowess", quietly = TRUE)) {
    stop(
        "rfastlowess is not installed. Build and install it first, e.g.\n",
        "  R CMD INSTALL .  (from bindings/r)\n",
        "or `make install`."
    )
}
suppressPackageStartupMessages(library("rfastlowess", character.only = TRUE))

source(file.path(testthat_dir, "helper-golden.R"))

cases <- golden_cases()

if (!dir.exists(fixture_dir)) {
    dir.create(fixture_dir, recursive = TRUE)
}

for (nm in names(cases)) {
    golden_write_csv(cases[[nm]], file.path(fixture_dir, paste0(nm, ".csv")))
}

files <- paste0(names(cases), ".csv")
sums <- vapply(
    files,
    function(f) unname(tools::md5sum(file.path(fixture_dir, f))),
    character(1)
)
provenance <- c(
    sprintf("generated_by: %s", "tests/testthat/fixtures/make_reference.R"),
    sprintf("generated_at: %s", format(Sys.time(), tz = "UTC", usetz = TRUE)),
    sprintf("r_version: %s", as.character(getRversion())),
    sprintf(
        "rfastlowess_version: %s",
        as.character(utils::packageVersion("rfastlowess"))
    ),
    sprintf("platform: %s", R.version$platform),
    sprintf("seed: %d", golden_seed()),
    sprintf("rng_kind: %s", paste(RNGkind(), collapse = " / ")),
    sprintf("tolerance: %.1e", 1e-10),
    "files:",
    sprintf("  %s: %s", files, unname(sums))
)
writeLines(provenance, file.path(fixture_dir, "PROVENANCE.txt"))

cat("Wrote", length(cases), "fixtures to", fixture_dir, "\n")
