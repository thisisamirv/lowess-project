#' Nullable Value Wrapper
#'
#' @description
#' Wraps a value to be passed to Rust as an Option.
#'
#' @param x Value to wrap or NULL.
#'
#' @return The value itself. This is a helper for rextendr conversion.
#' @examples
#' Nullable(5)
#' Nullable(NULL)
#' @export
Nullable <- function(x) {
    if (is.null(x)) {
        return(NULL)
    }
    x
}
