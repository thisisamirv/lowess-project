#!/usr/bin/env Rscript

# Custom style guide for 4-space indentation everywhere
style_4spaces <- function(...) {
    # Create base style with 4 spaces
    style <- styler::tidyverse_style(indent_by = 4, ...)
    
    # Patch unindent_function_declaration to use indent_by = 4
    # The default uses indent_by = 2L which overrides our preference
    if (!is.null(style$indention$unindent_function_declaration)) {
        old_fun <- style$indention$unindent_function_declaration
        style$indention$unindent_function_declaration <- function(
            pd,
            indent_by = 4
        ) {
            old_fun(pd, indent_by = indent_by)
        }
    }
    
    style
}

# Apply to the package
if (requireNamespace("styler", quietly = TRUE)) {
    message("Styling package with 4-space indentation...")
    styler::style_pkg(
        style = style_4spaces,
        filetype = c("R", "Rprofile", "Rmd", "Rmarkdown", "Rnw")
    )
} else {
    stop("Package 'styler' is required.")
}
