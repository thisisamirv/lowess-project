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

#' Stop with a Helpful Message if the Requested Backend is Unavailable
#' @noRd
check_gpu_backend <- function(backend) {
    if (identical(backend, "gpu") && !gpu_available()) {
        stop(
            "GPU backend not installed in this build. Run `install_gpu()` ",
            "once to download and install a GPU-enabled build, then ",
            "restart R. See https://thisisamirv.github.io/lowess-project/r/",
            "reference/gpu_available.html for details.",
            call. = FALSE
        )
    }
    invisible(NULL)
}

#' Determine the GPU Release Asset Name and Download URL
#' @noRd
gpu_asset_info <- function(version) {
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
    list(asset = asset, repo = repo, url = url, ext = ext)
}

# Wrappers for local_mocked_bindings(.package = "rfastlowess") in tests;
# base::interactive() is primitive and cannot be intercepted that way directly.
is_interactive <- function() interactive()
read_line <- function(prompt) readline(prompt)

#' Ask the User to Confirm the GPU Download, Unless Skipped
#' @noRd
gpu_confirm_download <- function(yes, asset, repo) {
    if (isTRUE(yes)) {
        return(invisible(TRUE))
    }
    if (!is_interactive()) {
        stop(
            "install_gpu() requires confirmation. Pass yes = TRUE to ",
            "proceed non-interactively.",
            call. = FALSE
        )
    }
    answer <- read_line(sprintf(
        "Download and install %s from github.com/%s? [y/N] ",
        asset,
        repo
    ))
    isTRUE(tolower(trimws(answer)) %in% c("y", "yes"))
}

#' Download the GPU Library to its Destination Path
#' @noRd
gpu_download_to <- function(url, ext, dest) {
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
#' # Check whether the GPU backend is already active before installing it
#' gpu_available()
#' if (interactive()) {
#'     install_gpu()
#' }
#' @export
install_gpu <- function(yes = FALSE) {
    if (gpu_available()) {
        message("GPU backend is already active.")
        return(invisible(TRUE))
    }

    version <- as.character(utils::packageVersion("rfastlowess"))
    info <- gpu_asset_info(version)

    if (!gpu_confirm_download(yes, info$asset, info$repo)) {
        message("Aborted.")
        return(invisible(FALSE))
    }

    lib_dir <- system.file("libs", package = "rfastlowess")
    if (identical(.Platform$OS.type, "windows") && nzchar(.Platform$r_arch)) {
        lib_dir <- file.path(lib_dir, .Platform$r_arch)
    }
    dest <- file.path(lib_dir, paste0("rfastlowess", info$ext))

    gpu_download_to(info$url, info$ext, dest)

    message("GPU backend installed at ", dest, ".")
    message("Restart R for the change to take effect.")
    invisible(TRUE)
}
