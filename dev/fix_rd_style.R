#!/usr/bin/env Rscript
# Fix indentation and wrap long author/seealso sections in .Rd files.
# Usage: Rscript fix_rd_style.R <man_dir>
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0L) stop("Usage: fix_rd_style.R <man_dir>", call. = FALSE)
man_dir <- args[[1L]]
bs <- rawToChar(as.raw(92L))
for (f in list.files(man_dir, "[.]Rd$", full.names = TRUE)) {
    orig <- readLines(f, warn = FALSE)
    ib <- FALSE
    out <- character(0L)
    for (l in orig) {
        n <- nchar(l) - nchar(trimws(l, "left"))
        if (n > 0L && n %% 4L != 0L) {
            l <- paste0(strrep(" ", ceiling(n / 4L) * 4L), trimws(l, "left"))
        }
        if (startsWith(l, paste0(bs, "author{")) || startsWith(l, paste0(bs, "seealso{"))) ib <- TRUE
        if (ib && l == "}") ib <- FALSE
        out <- c(out, if (ib && nchar(l) > 80L) strwrap(l, 80L, exdent = 4L) else l)
    }
    if (!identical(orig, out)) {
        writeLines(out, f)
        message("Fixed: ", basename(f))
    }
}
