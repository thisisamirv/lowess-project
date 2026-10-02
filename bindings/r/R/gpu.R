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

#' Probe a Candidate R Shared Library for the GPU Feature
#' @noRd
gpu_library_enabled <- function(path) {
    if (!file.exists(path)) {
        return(FALSE)
    }
    if (
        identical(Sys.info()[["sysname"]], "Windows") &&
            !identical(readBin(path, "raw", n = 2L), as.raw(c(0x4d, 0x5a)))
    ) {
        return(FALSE)
    }

    ext <- if (identical(Sys.info()[["sysname"]], "Windows")) ".dll" else ".so"
    probe_dir <- tempfile("rfastlowess-gpu-probe-")
    if (!dir.create(probe_dir)) {
        return(FALSE)
    }
    on.exit(unlink(probe_dir, recursive = TRUE), add = TRUE)
    probe_path <- file.path(probe_dir, paste0("rfastlowess", ext))
    if (!file.copy(path, probe_path, overwrite = TRUE)) {
        return(FALSE)
    }

    rscript <- file.path(
        R.home("bin"),
        if (identical(.Platform$OS.type, "windows")) "Rscript.exe" else "Rscript"
    )
    script <- paste(
        "probe <- function(path) {",
        "dll <- dyn.load(path)",
        "on.exit(dyn.unload(dll[['path']]), add = TRUE)",
        "routine <- getDLLRegisteredRoutines(dll)[['.Call']][['wrap__gpu_enabled']]",
        "if (is.null(routine)) return(FALSE)",
        "isTRUE(.Call(routine))",
        "}",
        "cat(probe(commandArgs(TRUE)[[1L]]))",
        sep = "\n"
    )
    output <- tryCatch(
        suppressWarnings(system2(
            rscript,
            args = c("--vanilla", "-e", shQuote(script), shQuote(probe_path)),
            stdout = TRUE,
            stderr = FALSE
        )),
        error = function(e) character()
    )
    status <- attr(output, "status")
    (is.null(status) || status == 0L) &&
        identical(trimws(paste(output, collapse = "")), "TRUE")
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
gpu_asset_info <- function(
    version,
    sys_name = Sys.info()[["sysname"]],
    machine = Sys.info()[["machine"]],
    r_platform = R.version$platform
) {
    if (identical(sys_name, "Windows")) {
        if (!grepl("^(x86[-_]64|amd64)$", machine, ignore.case = TRUE)) {
            stop(
                "No prebuilt R GPU library is available for this Windows architecture.",
                call. = FALSE
            )
        }
        platform_tag <- "windows"
        ext <- ".dll"
    } else if (identical(sys_name, "Darwin")) {
        if (!grepl("^(x86[-_]64|amd64|arm64|aarch64)$", machine, ignore.case = TRUE)) {
            stop("No prebuilt R GPU library is available for this macOS architecture.", call. = FALSE)
        }
        platform_tag <- "macos"
        # R uses .so as the package shared-object extension on macOS too
        ext <- ".so"
    } else if (identical(sys_name, "Linux")) {
        if (grepl("musl", r_platform, ignore.case = TRUE)) {
            stop("Prebuilt R GPU libraries are not available for musl Linux.", call. = FALSE)
        }
        if (!grepl("linux.*gnu", r_platform, ignore.case = TRUE)) {
            stop("Prebuilt R GPU libraries are available only for glibc Linux.", call. = FALSE)
        }
        if (!grepl("^(x86[-_]64|amd64)$", machine, ignore.case = TRUE)) {
            stop("Prebuilt R GPU libraries are available only for Linux x86_64.", call. = FALSE)
        }
        platform_tag <- "linux"
        ext <- ".so"
    } else {
        stop("No prebuilt R GPU library is available for this operating system.", call. = FALSE)
    }

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
    # GPU artifacts across all versions live in this one perpetual release
    # instead of cluttering each version's own release page; the source
    # version is embedded in the asset filename above instead.
    gpu_release_tag <- "gpu-builds"
    url <- sprintf(
        "https://github.com/%s/releases/download/%s/%s",
        repo,
        gpu_release_tag,
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

#' Atomically Replace an Installed Shared Library
#'
#' A running R session may still have the current library memory-mapped;
#' truncating/rewriting that file in place can segfault later. Refuse to fall
#' back to an in-place copy when atomic replacement is unavailable.
#' @noRd
gpu_replace_file <- function(src, dest) {
    tmp <- tempfile(
        tmpdir = dirname(dest),
        fileext = paste0(".", tools::file_ext(dest))
    )
    on.exit(unlink(tmp), add = TRUE)
    if (!file.copy(src, tmp, overwrite = TRUE)) {
        stop("Failed to stage install to ", dirname(dest), ".", call. = FALSE)
    }
    if (!file.rename(tmp, dest)) {
        stop(
            "Failed to atomically replace ", dest,
            "; refusing an in-place copy that could corrupt a loaded shared library.",
            call. = FALSE
        )
    }
    invisible(TRUE)
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
    if (!gpu_library_enabled(tmp)) {
        stop("Downloaded library does not report GPU support.", call. = FALSE)
    }
    gpu_replace_file(tmp, dest)
}

#' Resolve the Destination Directory for the Installed Shared Library
#' @noRd
gpu_lib_dir <- function(
    os_type = .Platform$OS.type,
    r_arch = .Platform$r_arch
) {
    lib_dir <- system.file("libs", package = "rfastlowess")
    if (identical(os_type, "windows") && nzchar(r_arch)) {
        lib_dir <- file.path(lib_dir, r_arch)
    }
    lib_dir
}

#' Ask the User to Confirm a Local-Path Install, Unless Skipped
#' @noRd
gpu_confirm_local_install <- function(yes, local_path) {
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
        "Install %s in place of the current build? [y/N] ",
        local_path
    ))
    isTRUE(tolower(trimws(answer)) %in% c("y", "yes"))
}

#' Install a GPU-Enabled Library Already Built Locally
#' @noRd
install_gpu_local <- function(local_path, yes, lib_dir) {
    if (!file.exists(local_path)) {
        stop("No such file: ", local_path, call. = FALSE)
    }
    if (!gpu_library_enabled(local_path)) {
        stop("The library at ", local_path, " does not report GPU support.", call. = FALSE)
    }
    if (!gpu_confirm_local_install(yes, local_path)) {
        message("Aborted.")
        return(invisible(FALSE))
    }

    ext <- paste0(".", tools::file_ext(local_path))
    dest <- file.path(lib_dir, paste0("rfastlowess", ext))
    message("Installing ", local_path, " ...")
    gpu_replace_file(local_path, dest)
    message("GPU backend installed at ", dest, ".")
    message("Restart R for the change to take effect.")
    invisible(TRUE)
}

#' Download and Install a GPU-Enabled Library from the Matching GitHub Release
#' @noRd
install_gpu_download <- function(yes, lib_dir) {
    version <- as.character(utils::packageVersion("rfastlowess"))
    info <- gpu_asset_info(version)

    if (!gpu_confirm_download(yes, info$asset, info$repo)) {
        message("Aborted.")
        return(invisible(FALSE))
    }

    dest <- file.path(lib_dir, paste0("rfastlowess", info$ext))
    gpu_download_to(info$url, info$ext, dest)
    message("GPU backend installed at ", dest, ".")
    message("Restart R for the change to take effect.")
    invisible(TRUE)
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
#' @param local_path Character; path to a GPU-enabled shared library already
#'   built locally (e.g. via \code{WITH_GPU=1 make install} in
#'   \code{benchmarks/}). When given, skips the GitHub Release lookup/download
#'   and installs directly from this path — useful for testing the installer
#'   itself, or installing an unreleased build.
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
install_gpu <- function(yes = FALSE, local_path = NULL) {
    if (gpu_available()) {
        message("GPU backend is already active.")
        return(invisible(TRUE))
    }

    lib_dir <- gpu_lib_dir()

    if (!is.null(local_path)) {
        return(install_gpu_local(local_path, yes, lib_dir))
    }

    install_gpu_download(yes, lib_dir)
}
