#' Check GPU Backend Availability
#'
#' @description
#' Returns whether the currently loaded \pkg{rfastlowess} shared library was
#' built with the GPU backend enabled.
#'
#' @return Logical; \code{TRUE} if the GPU backend is active.
#' @seealso \code{\link{install_gpu}} to download and install it.
#' @examples
#' gpu_available()
#' @export
gpu_available <- function() {
    isTRUE(gpu_enabled())
}

#' Download and Install the GPU-Enabled Backend
#'
#' @description
#' Downloads a prebuilt GPU-enabled \pkg{rfastlowess} shared library for the
#' current platform from the matching GitHub Release and installs it in
#' place of the current (CPU-only) library. GPU support is opt-in and not
#' included in CRAN/Bioconductor releases.
#'
#' A running R session cannot swap an already-loaded shared library, so
#' \strong{restart R} after installing for the change to take effect.
#'
#' @param yes Logical; skip the interactive \verb{y/N} confirmation prompt.
#'   Must be \code{TRUE} when the session is not interactive.
#' @return Invisibly, \code{TRUE} if a GPU-enabled library is available at
#'   the printed path (already active, or freshly installed); \code{FALSE}
#'   if the user aborted.
#' @seealso \code{\link{gpu_available}} to check the current status.
#' @examples
#' \dontrun{
#' install_gpu()
#' }
#' @export
install_gpu <- function(yes = FALSE) {
    if (gpu_available()) {
        message("GPU backend is already active.")
        return(invisible(TRUE))
    }

    version <- as.character(utils::packageVersion("rfastlowess"))
    sys_name <- Sys.info()[["sysname"]]

    if (identical(sys_name, "Windows")) {
        platform_tag <- "windows"
        ext <- ".dll"
    } else if (identical(sys_name, "Darwin")) {
        platform_tag <- "macos"
        # R uses .so as the package shared-object extension on macOS too
        ext <- ".so"
    } else {
        platform_tag <- "linux"
        ext <- ".so"
    }

    machine <- Sys.info()[["machine"]]
    is_arm <- grepl("arm|aarch64", machine, ignore.case = TRUE)
    arch <- if (is_arm) "aarch64" else "x86_64"

    asset <- sprintf(
        "librfastlowess-gpu-v%s-%s-%s%s",
        version,
        platform_tag,
        arch,
        ext
    )
    repo <- "thisisamirv/lowess-project"
    url <- sprintf(
        "https://github.com/%s/releases/download/v%s/%s",
        repo,
        version,
        asset
    )

    if (!isTRUE(yes)) {
        if (!interactive()) {
            stop(
                "install_gpu() requires confirmation. Pass yes = TRUE to ",
                "proceed non-interactively.",
                call. = FALSE
            )
        }
        answer <- readline(sprintf(
            "Download and install %s from github.com/%s? [y/N] ",
            asset,
            repo
        ))
        if (!tolower(trimws(answer)) %in% c("y", "yes")) {
            message("Aborted.")
            return(invisible(FALSE))
        }
    }

    lib_dir <- system.file("libs", package = "rfastlowess")
    if (identical(.Platform$OS.type, "windows") && nzchar(.Platform$r_arch)) {
        lib_dir <- file.path(lib_dir, .Platform$r_arch)
    }
    dest <- file.path(lib_dir, paste0("rfastlowess", ext))

    message("Downloading ", url, " ...")
    tmp <- tempfile(fileext = ext)
    on.exit(unlink(tmp), add = TRUE)
    ok <- tryCatch(
        {
            utils::download.file(url, tmp, mode = "wb", quiet = FALSE)
            TRUE
        },
        error = function(e) FALSE
    )
    if (!ok || !file.exists(tmp) || file.size(tmp) == 0) {
        stop(
            "Failed to download ",
            url,
            ".\n",
            "A matching GPU build may not exist for this platform/version yet.",
            call. = FALSE
        )
    }

    file.copy(tmp, dest, overwrite = TRUE)

    message("GPU backend installed at ", dest, ".")
    message("Restart R for the change to take effect.")
    invisible(TRUE)
}
