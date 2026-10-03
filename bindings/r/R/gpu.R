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
gpu_native_api <- function(namespace = asNamespace("rfastlowess")) {
    symbols <- ls(namespace, pattern = "^wrap__", all.names = TRUE)
    signatures <- vapply(symbols, function(name) {
        get(name, envir = namespace, inherits = FALSE)$numParameters
    }, integer(1))
    list(
        contract = paste0("rfastlowess/", getNamespaceVersion("rfastlowess"), "/abi-1"),
        signatures = signatures
    )
}

gpu_compatible_routines <- function(routines, expected) {
    required <- names(expected$signatures)
    if (!length(required) || !all(required %in% names(routines))) {
        return(FALSE)
    }
    counts <- vapply(routines[required], function(routine) routine$numParameters, integer(1))
    if (!identical(unname(counts), unname(expected$signatures))) {
        return(FALSE)
    }
    contract <- routines[["wrap__binding_contract"]]
    !is.null(contract) && identical(.Call(contract), expected$contract)
}

gpu_library_enabled <- function(path, timeout = 30L) {
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
    expected_path <- file.path(probe_dir, "api.rds")
    saveRDS(gpu_native_api(), expected_path)

    rscript <- file.path(
        R.home("bin"),
        if (identical(.Platform$OS.type, "windows")) "Rscript.exe" else "Rscript"
    )
    script <- paste(
        "probe <- function(path, expected) {",
        "dll <- dyn.load(path)",
        "on.exit(dyn.unload(dll[['path']]), add = TRUE)",
        "routines <- getDLLRegisteredRoutines(dll)[['.Call']]",
        "required <- names(expected$signatures)",
        "if (!all(required %in% names(routines))) return(FALSE)",
        "counts <- vapply(routines[required], function(routine) routine$numParameters, integer(1))",
        "if (!identical(unname(counts), unname(expected$signatures))) return(FALSE)",
        "contract <- routines[['wrap__binding_contract']]",
        "if (is.null(contract)) return(FALSE)",
        "if (!identical(.Call(contract), expected$contract)) return(FALSE)",
        "routine <- routines[['wrap__gpu_enabled']]",
        "!is.null(routine) && isTRUE(.Call(routine))",
        "}",
        "args <- commandArgs(TRUE)",
        "cat(probe(args[[1L]], readRDS(args[[2L]])))",
        sep = "\n"
    )
    script_path <- file.path(probe_dir, "probe.R")
    writeLines(script, script_path)
    output <- tryCatch(
        suppressWarnings(system2(
            rscript,
            args = c("--vanilla", shQuote(script_path), shQuote(probe_path), shQuote(expected_path)),
            stdout = TRUE,
            stderr = FALSE,
            timeout = timeout
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

gpu_is_musl <- function(
    r_platform = R.version$platform,
    sys_name = Sys.info()[["sysname"]]
) {
    if (!identical(sys_name, "Linux")) {
        return(FALSE)
    }
    if (grepl("musl", r_platform, ignore.case = TRUE)) {
        return(TRUE)
    }
    ldd <- Sys.which("ldd")
    output <- if (nzchar(ldd)) {
        tryCatch(
            suppressWarnings(system2(
                ldd,
                shQuote(file.path(R.home("bin"), "exec", "R")),
                stdout = TRUE,
                stderr = TRUE,
                timeout = 10L
            )),
            error = function(e) character()
        )
    } else {
        character()
    }
    if (any(grepl("ld-musl|libc[.]musl", output, ignore.case = TRUE))) {
        return(TRUE)
    }
    if (any(grepl("libc[.]so[.]6|ld-linux", output))) {
        return(FALSE)
    }
    stop(
        "Cannot determine the libc used by R; ",
        "install a locally built GPU library instead.",
        call. = FALSE
    )
}

#' Determine the GPU Release Asset Name and Download URL
#' @noRd
gpu_asset_info <- function(
    version,
    sys_name = Sys.info()[["sysname"]],
    machine = R.version$arch,
    r_platform = R.version$platform,
    musl = gpu_is_musl(r_platform, sys_name)
) {
    if (identical(sys_name, "Windows")) {
        if (!grepl("^(x86[-_]64|amd64|arm64|aarch64)$", machine, ignore.case = TRUE)) {
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
        if (!grepl("^(x86[-_]64|amd64|arm64|aarch64)$", machine, ignore.case = TRUE)) {
            stop("No prebuilt R GPU library is available for this Linux architecture.", call. = FALSE)
        }
        if (!musl && !grepl("linux.*gnu", r_platform, ignore.case = TRUE)) {
            stop("Prebuilt R GPU libraries require glibc or musl Linux.", call. = FALSE)
        }
        platform_tag <- if (musl) "linux-musl" else "linux"
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

gpu_release_digest <- function(url) {
    prefix <- paste0(
        "https://github.com/thisisamirv/lowess-project/",
        "releases/download/gpu-builds/"
    )
    if (!startsWith(url, prefix)) {
        stop("GPU downloads must come from the project's gpu-builds release.", call. = FALSE)
    }
    release <- jsonlite::fromJSON(paste0(
        "https://api.github.com/repos/thisisamirv/lowess-project/",
        "releases/tags/gpu-builds"
    ))
    assets <- release$assets
    if (!is.data.frame(assets) || !all(c("name", "browser_download_url", "digest") %in% names(assets))) {
        stop("GPU release does not provide SHA-256 asset digests.", call. = FALSE)
    }
    asset <- assets[
        assets$name == substring(url, nchar(prefix) + 1L) &
            assets$browser_download_url == url, ,
        drop = FALSE
    ]
    if (nrow(asset) != 1L || is.na(asset$digest) ||
        !grepl("^sha256:[[:xdigit:]]{64}$", asset$digest)) {
        stop("No trusted SHA-256 digest is available for this GPU asset.", call. = FALSE)
    }
    tolower(substring(asset$digest, 8L))
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
    expected_digest <- gpu_release_digest(url)
    actual_digest <- digest::digest(file = tmp, algo = "sha256", serialize = FALSE)
    if (!identical(actual_digest, expected_digest)) {
        stop("Downloaded GPU library failed SHA-256 verification.", call. = FALSE)
    }
    if (!gpu_library_enabled(tmp)) {
        stop("Downloaded library does not report compatible GPU support.", call. = FALSE)
    }
    gpu_replace_file(tmp, dest)
}

#' Resolve the Destination Directory for the Installed Shared Library
#' @noRd
gpu_lib_dir <- function(
    os_type = .Platform$OS.type,
    r_arch = .Platform$r_arch,
    lib_dir = system.file("libs", package = "rfastlowess")
) {
    if (identical(os_type, "windows") && nzchar(r_arch)) {
        lib_dir <- file.path(lib_dir, r_arch)
    }
    lib_dir
}

gpu_install_destination <- function(
    lib_dir,
    os_type = .Platform$OS.type,
    version = as.character(getNamespaceVersion("rfastlowess"))
) {
    if (identical(os_type, "windows")) {
        sidecar_dir <- tempfile(paste0("gpu-v", version, "-"), tmpdir = lib_dir)
        if (!dir.create(sidecar_dir)) {
            stop("Failed to create GPU library directory in ", lib_dir, call. = FALSE)
        }
        return(file.path(sidecar_dir, "rfastlowess.dll"))
    }
    file.path(lib_dir, "rfastlowess.so")
}

gpu_load_sidecar <- function(
    lib_dir,
    namespace,
    version,
    os_type = .Platform$OS.type
) {
    if (!identical(os_type, "windows")) {
        return(NULL)
    }
    directories <- list.dirs(lib_dir, recursive = FALSE, full.names = TRUE)
    directories <- directories[
        startsWith(basename(directories), paste0("gpu-v", version, "-"))
    ]
    paths <- file.path(directories, "rfastlowess.dll")
    paths <- paths[file.exists(paths)]
    if (!length(paths)) {
        return(NULL)
    }
    required <- ls(namespace, pattern = "^wrap__", all.names = TRUE)
    expected <- gpu_native_api()
    paths <- paths[order(file.info(paths)$mtime, decreasing = TRUE)]
    for (path in paths) {
        if (!gpu_library_enabled(path)) {
            warning("Skipping incompatible GPU sidecar: ", path, call. = FALSE)
            next
        }
        dll <- tryCatch(dyn.load(path), error = function(e) NULL)
        if (is.null(dll)) {
            warning("Skipping unloadable GPU sidecar: ", path, call. = FALSE)
            next
        }
        routines <- getDLLRegisteredRoutines(dll)[[".Call"]]
        compatible <- tryCatch(
            length(required) > 0L && all(required %in% names(routines)) &&
                gpu_compatible_routines(routines, expected),
            error = function(e) FALSE
        )
        if (!compatible) {
            dyn.unload(dll[["path"]])
            warning("Skipping incompatible GPU sidecar: ", path, call. = FALSE)
            next
        }
        if (any(vapply(required, bindingIsLocked, logical(1), env = namespace))) {
            dyn.unload(dll[["path"]])
            stop("GPU routines must be activated during namespace startup.", call. = FALSE)
        }
        for (name in required) {
            assign(name, routines[[name]], envir = namespace)
        }
        return(dll[["path"]])
    }
    NULL
}

gpu_state <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
    namespace <- asNamespace(pkgname)
    gpu_state$dll_path <- tryCatch(
        gpu_load_sidecar(
            gpu_lib_dir(lib_dir = file.path(libname, pkgname, "libs")), namespace,
            as.character(getNamespaceVersion(namespace))
        ),
        error = function(e) {
            warning("GPU sidecar was not activated: ", conditionMessage(e), call. = FALSE)
            NULL
        }
    )
}

.onUnload <- function(libpath) {
    invisible(NULL)
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
    if (!gpu_confirm_local_install(yes, local_path)) {
        message("Aborted.")
        return(invisible(FALSE))
    }
    if (!gpu_library_enabled(local_path)) {
        stop("The library at ", local_path, " does not report compatible GPU support.", call. = FALSE)
    }

    dest <- gpu_install_destination(lib_dir)
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

    dest <- gpu_install_destination(lib_dir, version = version)
    gpu_download_to(info$url, info$ext, dest)
    message("GPU backend installed at ", dest, ".")
    message("Restart R for the change to take effect.")
    invisible(TRUE)
}

#' Download and Install the GPU-Enabled Backend
#'
#' @description
#' Downloads a prebuilt GPU-enabled \pkg{rfastlowess} shared library for the
#' current platform from the matching GitHub Release. On Windows, installs
#' a versioned sidecar alongside the CPU library and activates its native
#' routines when the package is loaded after restarting R. Other platforms
#' atomically replace the CPU library. GPU support is opt-in and not included
#' in CRAN/Bioconductor releases.
#'
#' A running R session cannot swap an already-loaded shared library, so
#' \strong{restart R} after installing for the change to take effect.
#'
#' Downloads are checked against GitHub's published SHA-256 digest before
#' execution. Native probes have a 30-second timeout and require matching
#' package-version/ABI contracts and registered argument counts. Older GPU
#' artifacts without this contract must be rebuilt. On Windows, invalid
#' sidecars are skipped and activated DLLs remain mapped until R exits to
#' protect live model finalizers. Probing is not a security sandbox; local
#' native libraries must be trusted.
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
