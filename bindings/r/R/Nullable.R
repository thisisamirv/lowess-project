#' Wrap a value to be passed to Rust as an Option
#' @noRd
Nullable <- function(x) {
    if (is.null(x)) {
        return(NULL)
    }
    x
}
