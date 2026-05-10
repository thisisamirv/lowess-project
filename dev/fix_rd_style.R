#!/usr/bin/env Rscript
# Post-processes all Rd files in bindings/r/man/ after devtools::document().
# Fixes two pkgcheck NOTEs:
#   1. Indentation must be multiples of 4 spaces (roxygen2 default is 2).
#   2. Lines in \author{} and \seealso{} > 80 chars (long ORCID href / URLs).

# nolint start: indentation_linter.
rd_dir <- "bindings/r/man"
rd_files <- list.files(rd_dir, pattern = "\\.Rd$", full.names = TRUE)

# ---------------------------------------------------------------------------
# Helper: fix indentation globally to multiples of 4
# ---------------------------------------------------------------------------
fix_indentation <- function(lines) {
    out <- character(length(lines))
    for (i in seq_along(lines)) {
        l <- lines[[i]]
        # count leading spaces
        leading <- nchar(l) - nchar(trimws(l, "left"))
        if (leading > 0L && (leading %% 4L) != 0L) {
            # round up to next multiple of 4
            new_leading <- ceiling(leading / 4L) * 4L
            l <- paste0(strrep(" ", new_leading), trimws(l, "left"))
        }
        out[[i]] <- l
    }
    out
}

# ---------------------------------------------------------------------------
# Helper: wrap lines > 80 chars in \author{} and \seealso{} blocks at spaces
# ---------------------------------------------------------------------------
wrap_long_lines <- function(lines, width = 80L) {
    in_block <- FALSE
    out <- character(0)
    for (l in lines) {
        if (grepl("^\\\\(author|seealso)\\{", l)) in_block <- TRUE
        if (in_block && grepl("^\\}", l)) in_block <- FALSE

        if (in_block && nchar(l) > width) {
            # try to break at the last space before the limit
            wrapped <- strwrap(l, width = width, exdent = 4L)
            out <- c(out, wrapped)
        } else {
            out <- c(out, l)
        }
    }
    out
}

# ---------------------------------------------------------------------------
# Process every Rd file
# ---------------------------------------------------------------------------
changed <- 0L
for (f in rd_files) {
    original <- readLines(f, warn = FALSE)
    fixed <- fix_indentation(original)
    fixed <- wrap_long_lines(fixed)
    if (!identical(original, fixed)) {
        writeLines(fixed, f)
        changed <- changed + 1L
        message("Fixed: ", basename(f))
    }
}
message(sprintf("Done. %d file(s) modified.", changed))
# nolint end
