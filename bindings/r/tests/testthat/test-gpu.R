#' @srrstats {G5.3} No NA/NaN in validated outputs.
#' @srrstats {G5.8, G5.8a} Edge condition tests for GPU backend helpers.
# Tests targeting uncovered lines in gpu.R:
#   gpu_available / gpu_enabled (extendr-wrappers.R:110)
#   check_gpu_backend (both branches)
#   gpu_asset_info (all three platform branches, via mocking)
#   gpu_confirm_download (both branches)
#   gpu_download_to (success and failure, via mocking)
#   gpu_replace_file (success and both failure branches, via mocking)
#   gpu_lib_dir (windows multi-arch and default branches)
#   gpu_confirm_local_install (both branches)
#   install_gpu_local (missing file, declined, and confirmed branches)
#   install_gpu (already-active, needs-confirmation, and local_path dispatch)

check_gpu_backend <- getFromNamespace("check_gpu_backend", "rfastlowess")
gpu_asset_info <- getFromNamespace("gpu_asset_info", "rfastlowess")
gpu_confirm_download <- getFromNamespace("gpu_confirm_download", "rfastlowess")
gpu_confirm_local_install <- getFromNamespace(
    "gpu_confirm_local_install",
    "rfastlowess"
)
gpu_download_to <- getFromNamespace("gpu_download_to", "rfastlowess")
gpu_replace_file <- getFromNamespace("gpu_replace_file", "rfastlowess")
gpu_lib_dir <- getFromNamespace("gpu_lib_dir", "rfastlowess")
install_gpu_local <- getFromNamespace("install_gpu_local", "rfastlowess")
read_line <- getFromNamespace("read_line", "rfastlowess")

# ── read_line ─────────────────────────────────────────────────────────────────

test_that("native probe helpers validate files and process results", {
    valid_file <- getFromNamespace("gpu_valid_library_file", "rfastlowess")
    rscript <- getFromNamespace("gpu_rscript", "rfastlowess")
    succeeded <- getFromNamespace("gpu_probe_succeeded", "rfastlowess")
    src <- tempfile()
    writeLines("invalid binary", src)
    on.exit(unlink(src), add = TRUE)
    expect_false(valid_file(paste0(src, ".missing"), "Linux"))
    expect_false(valid_file(src, "Windows"))
    expect_true(valid_file(src, "Linux"))
    expect_identical(basename(rscript("windows")), "Rscript.exe")
    expect_identical(basename(rscript("unix")), "Rscript")
    expect_true(succeeded(" TRUE "))
    expect_false(succeeded("FALSE"))
    expect_false(succeeded(structure("TRUE", status = 1L)))
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    probe <- getFromNamespace("gpu_library_enabled", "rfastlowess")
    expect_false(probe(paste0(src, ".missing")))
    expect_identical(probe(native$dll[["path"]]), gpu_available())
})

test_that("probe directory and staging failures reject candidates", {
    probe <- getFromNamespace("gpu_library_enabled", "rfastlowess")
    prepare <- getFromNamespace("gpu_prepare_probe", "rfastlowess")
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    testthat::local_mocked_bindings(
        dir.create = function(...) FALSE,
        file.copy = function(...) FALSE,
        .package = "base"
    )
    expect_false(probe(native$dll[["path"]]))
    expect_null(prepare(native$dll[["path"]], tempdir()))
})

test_that("probe preparation and subprocess errors fail closed", {
    probe <- getFromNamespace("gpu_library_enabled", "rfastlowess")
    run_probe <- getFromNamespace("gpu_run_probe", "rfastlowess")
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    testthat::local_mocked_bindings(
        gpu_prepare_probe = function(...) NULL,
        .package = "rfastlowess"
    )
    expect_false(probe(native$dll[["path"]]))
    testthat::local_mocked_bindings(
        system2 = function(...) stop("process launch failed"),
        .package = "base"
    )
    expect_identical(run_probe(list(), 1L), character())
})

test_that("platform helpers reject unsupported systems and libc", {
    expect_error(
        gpu_asset_info("1.0.0", "FreeBSD", "x86_64"),
        "operating system"
    )
    expect_error(
        gpu_asset_info("1.0.0", "Darwin", "ppc64"),
        "macOS architecture"
    )
    expect_error(
        gpu_asset_info("1.0.0", "Linux", "x86_64", "linux-other", musl = FALSE),
        "glibc or musl"
    )
})

test_that("release metadata without required digest fields is rejected", {
    release_digest <- getFromNamespace("gpu_release_digest", "rfastlowess")
    url <- paste0(
        "https://github.com/thisisamirv/lowess-project/",
        "releases/download/gpu-builds/library.so"
    )
    testthat::local_mocked_bindings(
        fromJSON = function(...) list(assets = data.frame(name = "library.so")),
        .package = "jsonlite"
    )
    expect_error(release_digest(url), "does not provide SHA-256")
})

test_that("Windows destination creation errors are reported", {
    destination <- getFromNamespace("gpu_install_destination", "rfastlowess")
    testthat::local_mocked_bindings(
        dir.create = function(...) FALSE,
        .package = "base"
    )
    expect_error(destination(tempdir(), "windows", "1.2.3"), "Failed to create")
})

test_that("locked namespaces are not partially rebound", {
    bind_sidecar <- getFromNamespace("gpu_bind_sidecar", "rfastlowess")
    namespace <- new.env(parent = emptyenv())
    namespace$wrap__gpu_enabled <- "original"
    lockBinding("wrap__gpu_enabled", namespace)
    unloaded <- new.env()
    testthat::local_mocked_bindings(
        dyn.unload = function(path) unloaded$path <- path,
        .package = "base"
    )
    expect_error(
        bind_sidecar(namespace, list(), list(path = "candidate.dll")),
        "namespace startup"
    )
    expect_identical(namespace$wrap__gpu_enabled, "original")
    expect_identical(unloaded$path, "candidate.dll")
})

test_that("incompatible sidecars are rejected before loading", {
    try_sidecar <- getFromNamespace("gpu_try_sidecar", "rfastlowess")
    namespace <- new.env(parent = emptyenv())
    namespace$wrap__gpu_enabled <- "original"
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(...) FALSE,
        .package = "rfastlowess"
    )
    expect_warning(
        expect_null(try_sidecar(
            "candidate.dll",
            namespace,
            list(),
            load_library = function(...) stop("Candidate must not load")
        )),
        "Skipping incompatible GPU sidecar"
    )
    expect_identical(namespace$wrap__gpu_enabled, "original")
})

test_that("unloadable sidecars are skipped", {
    try_sidecar <- getFromNamespace("gpu_try_sidecar", "rfastlowess")
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(...) TRUE,
        .package = "rfastlowess"
    )
    testthat::local_mocked_bindings(
        dyn.load = function(...) stop("loader failure"),
        .package = "base"
    )
    expect_warning(
        expect_null(try_sidecar("candidate.dll", new.env(), list())),
        "Skipping unloadable"
    )
})

test_that("startup errors warn and retain CPU fallback", {
    hook <- getFromNamespace(".onLoad", "rfastlowess")
    state <- getFromNamespace("gpu_state", "rfastlowess")
    previous <- state$dll_path
    on.exit(state$dll_path <- previous, add = TRUE)
    testthat::local_mocked_bindings(
        gpu_load_sidecar = function(lib_dir, namespace, version) {
            expect_type(lib_dir, "character")
            expect_type(namespace, "environment")
            expect_type(version, "character")
            stop("startup failure")
        },
        .package = "rfastlowess"
    )
    expect_warning(
        hook(tempdir(), "rfastlowess"),
        "GPU sidecar was not activated"
    )
    expect_null(state$dll_path)
})

test_that("unloading does not clear the retained GPU library path", {
    hook <- getFromNamespace(".onUnload", "rfastlowess")
    state <- getFromNamespace("gpu_state", "rfastlowess")
    previous <- state$dll_path
    on.exit(state$dll_path <- previous, add = TRUE)
    state$dll_path <- "candidate.dll"
    expect_null(hook(tempdir()))
    expect_identical(state$dll_path, "candidate.dll")
})

test_that("read_line delegates to readline", {
    testthat::local_mocked_bindings(
        readline = function(prompt) paste0("echo:", prompt),
        .package = "base"
    )
    expect_identical(read_line("Enter: "), "echo:Enter: ")
})

# ── gpu_available / gpu_enabled ──────────────────────────────────────────────

test_that("gpu_available returns a single logical", {
    result <- gpu_available()
    expect_type(result, "logical")
    expect_length(result, 1)
})

# ── check_gpu_backend ────────────────────────────────────────────────────────

test_that("check_gpu_backend allows any non-gpu backend", {
    expect_null(check_gpu_backend("cpu"))
    expect_null(check_gpu_backend(NULL))
    expect_null(check_gpu_backend("nonsense"))
})

test_that("check_gpu_backend errors for gpu backend when unavailable", {
    skip_if(gpu_available(), "GPU backend is active in this build")
    expect_error(
        check_gpu_backend("gpu"),
        "GPU backend not installed in this build"
    )
})

test_that("check_gpu_backend allows gpu backend when available", {
    testthat::local_mocked_bindings(gpu_available = function() TRUE)
    expect_null(check_gpu_backend("gpu"))
})

# ── gpu_asset_info ───────────────────────────────────────────────────────────

test_that("gpu_asset_info builds a well-formed asset name and URL", {
    info <- gpu_asset_info("1.2.3")
    expect_true(grepl("^librfastlowess-gpu-v1\\.2\\.3-", info$asset))
    expect_true(endsWith(info$asset, info$ext))
    expect_identical(info$repo, "thisisamirv/lowess-project")
    expect_identical(
        info$url,
        sprintf(
            "https://github.com/%s/releases/download/gpu-builds/%s",
            info$repo,
            info$asset
        )
    )
    expect_true(info$ext %in% c(".dll", ".so"))
})

test_that("gpu_asset_info handles Windows platform", {
    testthat::local_mocked_bindings(
        `Sys.info` = function() c(sysname = "Windows", machine = "x86-64"),
        .package = "base"
    )
    info <- gpu_asset_info("1.0.0", machine = "x86-64")
    expect_identical(info$ext, ".dll")
    expect_true(grepl("windows-x86_64\\.dll$", info$asset))
})

test_that("gpu_asset_info handles macOS arm64 platform", {
    testthat::local_mocked_bindings(
        `Sys.info` = function() c(sysname = "Darwin", machine = "arm64"),
        .package = "base"
    )
    info <- gpu_asset_info("1.0.0", machine = "arm64")
    expect_identical(info$ext, ".so")
    expect_true(grepl("macos-aarch64\\.so$", info$asset))
})

test_that("gpu_asset_info handles Linux platform", {
    testthat::local_mocked_bindings(
        `Sys.info` = function() c(sysname = "Linux", machine = "x86_64"),
        .package = "base"
    )
    info <- gpu_asset_info(
        "1.0.0",
        machine = "x86_64",
        r_platform = "x86_64-pc-linux-gnu",
        musl = FALSE
    )
    expect_identical(info$ext, ".so")
    expect_true(grepl("linux-x86_64\\.so$", info$asset))
})

test_that("gpu_asset_info selects ARM64 and musl R GPU targets", {
    targets <- list(
        c(
            "Linux",
            "x86_64",
            "x86_64-alpine-linux-musl",
            "linux-musl-x86_64.so"
        ),
        c(
            "Linux",
            "aarch64",
            "aarch64-alpine-linux-musl",
            "linux-musl-aarch64.so"
        ),
        c("Linux", "aarch64", "aarch64-unknown-linux-gnu", "linux-aarch64.so"),
        c("Windows", "ARM64", "aarch64-w64-mingw32", "windows-aarch64.dll")
    )
    for (target in targets) {
        info <- gpu_asset_info(
            "1.0.0",
            target[1],
            target[2],
            target[3],
            musl = grepl("musl", target[3], fixed = TRUE)
        )
        expect_identical(
            info$asset,
            paste0("librfastlowess-gpu-v1.0.0-", target[4])
        )
    }
    info <- gpu_asset_info(
        "1.0.0",
        "Linux",
        "x86_64",
        "x86_64-pc-linux-gnu",
        musl = TRUE
    )
    expect_true(endsWith(info$asset, "linux-musl-x86_64.so"))
})

test_that("gpu_asset_info rejects unsupported R GPU targets", {
    expect_error(
        gpu_asset_info(
            "1.0.0",
            "Linux",
            "riscv64",
            "riscv64-unknown-linux-gnu"
        ),
        "Linux architecture"
    )
    expect_error(
        gpu_asset_info("1.0.0", "Windows", "i686", "i686-w64-mingw32"),
        "Windows architecture"
    )
})

test_that("GPU libc detection follows R's linked runtime", {
    gpu_is_musl <- getFromNamespace("gpu_is_musl", "rfastlowess")
    testthat::local_mocked_bindings(
        Sys.which = function(...) "ldd",
        Sys.glob = function(...) "/lib/ld-musl-x86_64.so.1",
        system2 = function(...) "libc.so.6 => /lib/libc.so.6",
        .package = "base"
    )
    expect_false(gpu_is_musl("x86_64-pc-linux-gnu", "Linux"))
    info <- gpu_asset_info("1.0.0", "Linux", "x86_64", "x86_64-pc-linux-gnu")
    expect_true(endsWith(info$asset, "linux-x86_64.so"))
})

test_that("GPU libc detection recognizes musl and rejects unknown runtimes", {
    gpu_is_musl <- getFromNamespace("gpu_is_musl", "rfastlowess")
    testthat::local_mocked_bindings(
        Sys.which = function(...) "ldd",
        system2 = function(...) {
            "libc.musl-aarch64.so.1 => /lib/ld-musl-aarch64.so.1"
        },
        .package = "base"
    )
    expect_true(gpu_is_musl("aarch64-unknown-linux-gnu", "Linux"))
    expect_true(gpu_is_musl("aarch64-alpine-linux-musl", "Linux"))
    expect_false(gpu_is_musl("aarch64-w64-mingw32", "Windows"))
})

test_that("GPU libc detection does not guess when inspection fails", {
    gpu_is_musl <- getFromNamespace("gpu_is_musl", "rfastlowess")
    testthat::local_mocked_bindings(
        Sys.which = function(...) "",
        .package = "base"
    )
    expect_error(
        gpu_is_musl("x86_64-pc-linux-gnu", "Linux"),
        "Cannot determine"
    )
})

test_that("GPU destinations use canonical names and unique Windows sidecars", {
    destination <- getFromNamespace("gpu_install_destination", "rfastlowess")
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    expect_identical(
        destination(lib_dir, "unix", "1.2.3"),
        file.path(lib_dir, "rfastlowess.so")
    )
    first <- destination(lib_dir, "windows", "1.2.3")
    second <- destination(lib_dir, "windows", "1.2.3")
    expect_identical(basename(first), "rfastlowess.dll")
    expect_true(startsWith(basename(dirname(first)), "gpu-v1.2.3-"))
    expect_false(identical(first, second))
})

test_that("Windows installation preserves a loaded primary DLL", {
    skip_if_not(identical(.Platform$OS.type, "windows"))
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    primary <- file.path(lib_dir, "rfastlowess.dll")
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    file.copy(native$dll[["path"]], primary)
    dll <- dyn.load(primary)
    on.exit(dyn.unload(dll[["path"]]), add = TRUE)
    original <- readBin(primary, "raw", n = file.size(primary))
    source <- tempfile(fileext = ".renamed")
    writeLines("validated candidate", source)
    on.exit(unlink(source), add = TRUE)
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(...) TRUE,
        .package = "rfastlowess"
    )
    expect_true(install_gpu_local(source, TRUE, lib_dir))
    expect_identical(readBin(primary, "raw", n = file.size(primary)), original)
    installed <- list.files(lib_dir, recursive = TRUE, full.names = TRUE)
    expect_length(installed, 2L)
    expect_true(all(basename(installed) == "rfastlowess.dll"))
})

test_that("sidecar activation rebinds version-matched native routines", {
    loader <- getFromNamespace("gpu_load_sidecar", "rfastlowess")
    try_sidecar <- getFromNamespace("gpu_try_sidecar", "rfastlowess")
    native_api <- getFromNamespace("gpu_native_api", "rfastlowess")
    destination <- getFromNamespace("gpu_install_destination", "rfastlowess")
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    path <- destination(lib_dir, "windows", "1.2.3")
    file.copy(native$dll[["path"]], path)
    original_namespace <- asNamespace("rfastlowess")
    namespace <- new.env(parent = emptyenv())
    required <- ls(original_namespace, pattern = "^wrap__", all.names = TRUE)
    for (name in required) {
        assign(name, get(name, envir = original_namespace), envir = namespace)
    }
    routines <- getDLLRegisteredRoutines(native$dll)[[".Call"]]
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(...) TRUE,
        .package = "rfastlowess"
    )
    load_library <- function(candidate) list(path = candidate)
    registered_routines <- function(...) list(.Call = routines)
    unload_library <- function(...) invisible(NULL)
    activated <- try_sidecar(
        path,
        namespace,
        native_api(),
        load_library,
        registered_routines,
        unload_library
    )
    expect_identical(activated, path)
    expect_identical(
        namespace$wrap__gpu_enabled,
        routines[["wrap__gpu_enabled"]]
    )
    expect_identical(.Call(namespace$wrap__gpu_enabled), .Call(native))

    try_candidate <- function(candidate, ...) {
        if (identical(normalizePath(candidate), normalizePath(path))) {
            candidate
        } else {
            NULL
        }
    }
    testthat::local_mocked_bindings(
        gpu_try_sidecar = try_candidate,
        .package = "rfastlowess"
    )
    expect_null(loader(lib_dir, namespace, "9.9.9", "windows"))
    expect_null(loader(lib_dir, namespace, "1.2.3", "unix"))
    loaded <- loader(lib_dir, namespace, "1.2.3", "windows")
    expect_identical(normalizePath(loaded), normalizePath(path))
    injected <- loader(
        lib_dir,
        namespace,
        "1.2.3",
        "windows",
        try_candidate = try_candidate
    )
    expect_identical(normalizePath(injected), normalizePath(path))
})

test_that("incompatible sidecars leave existing native routines unchanged", {
    loader <- getFromNamespace("gpu_load_sidecar", "rfastlowess")
    destination <- getFromNamespace("gpu_install_destination", "rfastlowess")
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    path <- destination(lib_dir, "windows", "1.2.3")
    file.copy(native$dll[["path"]], path)
    namespace <- new.env(parent = emptyenv())
    namespace$wrap__missing_api <- "original"
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(...) TRUE,
        .package = "rfastlowess"
    )
    expect_warning(
        expect_null(loader(lib_dir, namespace, "1.2.3", "windows")),
        "incompatible"
    )
    expect_identical(namespace$wrap__missing_api, "original")
})

test_that("native compatibility checks enforce contract and argument counts", {
    api <- getFromNamespace("gpu_native_api", "rfastlowess")()
    compatible <- getFromNamespace("gpu_compatible_routines", "rfastlowess")
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    routines <- getDLLRegisteredRoutines(native$dll)[[".Call"]]
    expect_true(compatible(routines, api))
    wrong_arity <- api
    wrong_arity$signatures[1L] <- wrong_arity$signatures[1L] + 1L
    expect_false(compatible(routines, wrong_arity))
    wrong_contract <- api
    wrong_contract$contract <- "rfastlowess/0.0.0/abi-999"
    expect_false(compatible(routines, wrong_contract))
    expect_false(compatible(routines[-1L], api))
    wrong_candidate <- routines
    candidate_gpu <- routines[["wrap__gpu_enabled"]]
    wrong_candidate[["wrap__binding_contract"]] <- candidate_gpu
    expect_false(compatible(wrong_candidate, api))
})

test_that("candidate subprocesses have an enforced timeout", {
    probe <- getFromNamespace("gpu_library_enabled", "rfastlowess")
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    captured <- new.env()
    testthat::local_mocked_bindings(
        system2 = function(command, args, stdout, stderr, timeout) {
            captured$timeout <- timeout
            structure(character(), status = 124L)
        },
        .package = "base"
    )
    expect_false(probe(native$dll[["path"]], timeout = 2L))
    expect_identical(captured$timeout, 2L)
})

test_that("sidecar activation falls back from a broken newest candidate", {
    loader <- getFromNamespace("gpu_load_sidecar", "rfastlowess")
    destination <- getFromNamespace("gpu_install_destination", "rfastlowess")
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    native <- getFromNamespace("wrap__gpu_enabled", "rfastlowess")
    older <- destination(lib_dir, "windows", "1.2.3")
    file.copy(native$dll[["path"]], older)
    Sys.setFileTime(older, Sys.time() - 60)
    newer <- destination(lib_dir, "windows", "1.2.3")
    writeLines("broken", newer)
    testthat::local_mocked_bindings(
        gpu_try_sidecar = function(path, ...) {
            if (identical(normalizePath(path), normalizePath(older))) {
                return(path)
            }
            warning("Skipping incompatible GPU sidecar: ", path, call. = FALSE)
            NULL
        },
        .package = "rfastlowess"
    )
    expect_warning(
        loaded <- loader(lib_dir, new.env(), "1.2.3", "windows"),
        "Skipping incompatible"
    )
    expect_identical(normalizePath(loaded), normalizePath(older))
})

test_that("release metadata must supply an exact matching SHA-256 digest", {
    release_digest <- getFromNamespace("gpu_release_digest", "rfastlowess")
    url <- paste0(
        "https://github.com/thisisamirv/lowess-project/",
        "releases/download/gpu-builds/library.so"
    )
    checksum <- strrep("a", 64L)
    testthat::local_mocked_bindings(
        fromJSON = function(...) {
            list(
                assets = data.frame(
                    name = "library.so",
                    browser_download_url = url,
                    digest = paste0("sha256:", checksum)
                )
            )
        },
        .package = "jsonlite"
    )
    expect_identical(release_digest(url), checksum)
    expect_error(
        release_digest("https://example.com/library.so"),
        "must come from"
    )
    expect_error(release_digest(paste0(url, ".missing")), "No trusted SHA-256")
})

test_that("digest verification precedes downloaded native execution", {
    dest <- tempfile()
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        download.file = function(url, destfile, ...) {
            writeLines("tampered", destfile)
            0L
        },
        .package = "utils"
    )
    testthat::local_mocked_bindings(
        gpu_release_digest = function(...) strrep("0", 64L),
        gpu_library_enabled = function(...) stop("Must not execute candidate"),
        .package = "rfastlowess"
    )
    expect_error(
        gpu_download_to("https://example.com/lib.so", ".so", dest),
        "SHA-256 verification"
    )
    expect_false(file.exists(dest))
})

test_that("namespace unload keeps live model finalizers mapped", {
    script <- tempfile(fileext = ".R")
    on.exit(unlink(script), add = TRUE)
    writeLines(
        c(
            "package_path <- commandArgs(TRUE)[1L]",
            "if (file.exists(file.path(package_path, 'R', 'gpu.R'))) {",
            "pkgload::load_all(package_path, quiet = TRUE)",
            "} else {",
            "library('rfastlowess', lib.loc = dirname(package_path))",
            "}",
            "ns <- asNamespace('rfastlowess')",
            "filename <- paste0('rfastlowess', .Platform$dynlib.ext)",
            "path <- file.path(tempdir(), filename)",
            "file.copy(ns$wrap__gpu_enabled$dll[['path']], path)",
            "dll <- dyn.load(path)",
            "scope <- new.env(parent = ns)",
            "scope$RLowess <- new.env()",
            "factory <- ns$RLowess$new",
            "environment(factory) <- scope",
            "scope$RLowess$new <- factory",
            "routines <- getDLLRegisteredRoutines(dll)[['.Call']]",
            "scope$wrap__RLowess__new <- routines[['wrap__RLowess__new']]",
            "constructor <- ns$Lowess",
            "environment(constructor) <- scope",
            "model <- constructor()",
            "state <- ns$gpu_state",
            "state$dll_path <- dll[['path']]",
            "ns$.onUnload('unused')",
            "rm(model)",
            "gc()",
            "loaded_paths <- vapply(",
            "getLoadedDLLs(), function(info) info[['path']], ''",
            ")",
            "stopifnot(dll[['path']] %in% loaded_paths)",
            "cat('FINALIZERS_SAFE')"
        ),
        script
    )
    rscript <- file.path(
        R.home("bin"),
        if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript"
    )
    output <- system2(
        rscript,
        c(
            "--vanilla",
            shQuote(script),
            shQuote(getNamespaceInfo("rfastlowess", "path"))
        ),
        stdout = TRUE,
        stderr = TRUE,
        timeout = 60L
    )
    expect_null(
        attr(output, "status"),
        info = paste(output, collapse = "\n")
    )
    expect_true(any(grepl("FINALIZERS_SAFE", output, fixed = TRUE)))
})

# ── gpu_confirm_download ─────────────────────────────────────────────────────

test_that("gpu_confirm_download returns TRUE when yes = TRUE", {
    expect_true(gpu_confirm_download(TRUE, "asset", "repo"))
})

test_that("gpu_confirm_download errors non-interactively", {
    expect_error(
        gpu_confirm_download(FALSE, "asset", "repo"),
        "install_gpu\\(\\) requires confirmation"
    )
})

test_that("gpu_confirm_download accepts an interactive 'y' answer", {
    testthat::local_mocked_bindings(
        is_interactive = function() TRUE,
        read_line = function(prompt) "y",
        .package = "rfastlowess"
    )
    expect_true(gpu_confirm_download(FALSE, "asset", "repo"))
})

test_that("gpu_confirm_download declines a non-'y' interactive answer", {
    testthat::local_mocked_bindings(
        is_interactive = function() TRUE,
        read_line = function(prompt) "n",
        .package = "rfastlowess"
    )
    expect_false(gpu_confirm_download(FALSE, "asset", "repo"))
})

# ── gpu_download_to ──────────────────────────────────────────────────────────

test_that("gpu_download_to copies the downloaded file to its destination", {
    dest <- tempfile()
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        download.file = function(url, destfile, mode, quiet) {
            writeLines("dummy", destfile)
            0L
        },
        .package = "utils"
    )
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(path) TRUE,
        gpu_release_digest = function(url) {
            fixture <- tempfile()
            writeLines("dummy", fixture)
            on.exit(unlink(fixture), add = TRUE)
            digest::digest(file = fixture, algo = "sha256", serialize = FALSE)
        },
        .package = "rfastlowess"
    )
    gpu_download_to("https://example.com/lib.so", ".so", dest)
    expect_true(file.exists(dest))
    expect_gt(file.size(dest), 0)
})

test_that("gpu_download_to rejects a library without the GPU feature", {
    dest <- tempfile(fileext = ".so")
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        download.file = function(url, destfile, mode, quiet) {
            writeLines("not a shared library", destfile)
            0L
        },
        .package = "utils"
    )
    testthat::local_mocked_bindings(
        gpu_release_digest = function(url) {
            fixture <- tempfile()
            writeLines("not a shared library", fixture)
            on.exit(unlink(fixture), add = TRUE)
            digest::digest(file = fixture, algo = "sha256", serialize = FALSE)
        },
        .package = "rfastlowess"
    )
    expect_error(
        gpu_download_to("https://example.com/lib.so", ".so", dest),
        "does not report compatible GPU support"
    )
    expect_false(file.exists(dest))
})

test_that("gpu_download_to errors when the download fails", {
    dest <- tempfile()
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        download.file = function(url, destfile, mode, quiet) {
            stop("network unreachable")
        },
        .package = "utils"
    )
    expect_error(
        gpu_download_to("https://example.com/lib.so", ".so", dest),
        "Failed to download"
    )
})

test_that("gpu_download_to errors when the downloaded file is empty", {
    dest <- tempfile()
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        download.file = function(url, destfile, mode, quiet) {
            file.create(destfile)
            0L
        },
        .package = "utils"
    )
    expect_error(
        gpu_download_to("https://example.com/lib.so", ".so", dest),
        "Failed to download"
    )
})

# ── gpu_replace_file ──────────────────────────────────────────────────────────

test_that("gpu_replace_file errors when staging the copy fails", {
    src <- tempfile()
    writeLines("dummy", src)
    on.exit(unlink(src), add = TRUE)
    dest <- tempfile()
    testthat::local_mocked_bindings(
        `file.copy` = function(from, to, overwrite = FALSE) FALSE,
        .package = "base"
    )
    expect_error(gpu_replace_file(src, dest), "Failed to stage install")
})

test_that("gpu_replace_file refuses an in-place overwrite when rename fails", {
    src <- tempfile()
    writeLines("replacement", src)
    on.exit(unlink(src), add = TRUE)
    dest <- tempfile()
    writeLines("original", dest)
    on.exit(unlink(dest), add = TRUE)
    testthat::local_mocked_bindings(
        `file.rename` = function(from, to) FALSE,
        `file.copy` = function(from, to, overwrite = FALSE) {
            !identical(to, dest)
        },
        .package = "base"
    )
    expect_error(gpu_replace_file(src, dest), "refusing an in-place copy")
    expect_identical(readLines(dest), "original")
})

# ── install_gpu ──────────────────────────────────────────────────────────────

test_that("gpu_lib_dir appends r_arch on windows multi-arch installs", {
    base_dir <- system.file("libs", package = "rfastlowess")
    expect_identical(
        gpu_lib_dir(os_type = "windows", r_arch = "x64"),
        file.path(base_dir, "x64")
    )
})

test_that("gpu_lib_dir skips r_arch subdir off windows or without r_arch", {
    base_dir <- system.file("libs", package = "rfastlowess")
    expect_identical(gpu_lib_dir(os_type = "unix", r_arch = "x64"), base_dir)
    expect_identical(gpu_lib_dir(os_type = "windows", r_arch = ""), base_dir)
})

test_that("install_gpu short-circuits when the backend is already active", {
    testthat::local_mocked_bindings(gpu_available = function() TRUE)
    expect_message(
        result <- install_gpu(),
        "GPU backend is already active"
    )
    expect_true(isTRUE(result))
})

test_that("install_gpu requires confirmation non-interactively when inactive", {
    skip_if(gpu_available(), "GPU backend is active in this build")
    expect_error(
        install_gpu(yes = FALSE),
        "install_gpu\\(\\) requires confirmation"
    )
})

test_that("install_gpu aborts when confirmation is declined", {
    skip_if(gpu_available(), "GPU backend is active in this build")
    testthat::local_mocked_bindings(
        gpu_confirm_download = function(yes, asset, repo) FALSE
    )
    expect_message(
        result <- install_gpu(yes = FALSE),
        "Aborted"
    )
    expect_false(isTRUE(result))
})

test_that("install_gpu downloads and installs when confirmed", {
    skip_if(gpu_available(), "GPU backend is active in this build")
    testthat::local_mocked_bindings(
        gpu_confirm_download = function(yes, asset, repo) TRUE,
        gpu_install_destination = function(...) "mock-destination",
        gpu_download_to = function(url, ext, dest) invisible(TRUE)
    )
    expect_message(
        result <- install_gpu(yes = TRUE),
        "GPU backend installed at"
    )
    expect_true(isTRUE(result))
})

# ── gpu_confirm_local_install ─────────────────────────────────────────────────

test_that("gpu_confirm_local_install returns TRUE when yes = TRUE", {
    expect_true(gpu_confirm_local_install(TRUE, "/tmp/lib.so"))
})

test_that("gpu_confirm_local_install errors non-interactively", {
    expect_error(
        gpu_confirm_local_install(FALSE, "/tmp/lib.so"),
        "install_gpu\\(\\) requires confirmation"
    )
})

test_that("gpu_confirm_local_install accepts an interactive 'y' answer", {
    testthat::local_mocked_bindings(
        is_interactive = function() TRUE,
        read_line = function(prompt) "y",
        .package = "rfastlowess"
    )
    expect_true(gpu_confirm_local_install(FALSE, "/tmp/lib.so"))
})

test_that("gpu_confirm_local_install declines a non-'y' interactive answer", {
    testthat::local_mocked_bindings(
        is_interactive = function() TRUE,
        read_line = function(prompt) "n",
        .package = "rfastlowess"
    )
    expect_false(gpu_confirm_local_install(FALSE, "/tmp/lib.so"))
})

# ── install_gpu_local ─────────────────────────────────────────────────────────

test_that("install_gpu_local errors when the file does not exist", {
    expect_error(
        install_gpu_local("/no/such/file.so", TRUE, tempdir()),
        "No such file"
    )
})

test_that("install_gpu_local aborts when confirmation is declined", {
    src <- tempfile(fileext = ".so")
    writeLines("dummy", src)
    on.exit(unlink(src), add = TRUE)
    testthat::local_mocked_bindings(
        gpu_confirm_local_install = function(yes, local_path) FALSE,
        gpu_library_enabled = function(path) {
            stop("Must not probe before confirmation")
        }
    )
    expect_message(
        result <- install_gpu_local(src, FALSE, tempdir()),
        "Aborted"
    )
    expect_false(isTRUE(result))
})

test_that("non-interactive local installs require consent before probing", {
    src <- tempfile()
    file.create(src)
    on.exit(unlink(src), add = TRUE)
    testthat::local_mocked_bindings(
        is_interactive = function() FALSE,
        gpu_library_enabled = function(...) stop("Candidate must not execute"),
        .package = "rfastlowess"
    )
    expect_error(
        install_gpu_local(src, FALSE, tempdir()),
        "requires confirmation"
    )
})

test_that("install_gpu_local rejects non-GPU files", {
    src <- tempfile(fileext = ".so")
    writeLines("dummy", src)
    on.exit(unlink(src), add = TRUE)
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)

    expect_error(
        install_gpu_local(src, TRUE, lib_dir),
        "does not report compatible GPU support"
    )
    expect_false(file.exists(file.path(lib_dir, "rfastlowess.so")))
})

test_that("install_gpu_local installs a validated GPU library", {
    src <- tempfile(fileext = ".so")
    writeLines("gpu library", src)
    on.exit(unlink(src), add = TRUE)
    lib_dir <- tempfile()
    dir.create(lib_dir)
    on.exit(unlink(lib_dir, recursive = TRUE), add = TRUE)
    testthat::local_mocked_bindings(
        gpu_library_enabled = function(path) TRUE,
        .package = "rfastlowess"
    )

    expect_message(
        result <- install_gpu_local(src, TRUE, lib_dir),
        "GPU backend installed at"
    )
    expect_true(isTRUE(result))
    installed <- list.files(lib_dir, recursive = TRUE, full.names = TRUE)
    expect_length(installed, 1L)
    expect_identical(
        basename(installed),
        paste0("rfastlowess", .Platform$dynlib.ext)
    )
})

test_that("install_gpu dispatches to install_gpu_local for local_path", {
    skip_if(gpu_available(), "GPU backend is active in this build")
    src <- tempfile(fileext = ".so")
    writeLines("dummy", src)
    on.exit(unlink(src), add = TRUE)
    testthat::local_mocked_bindings(
        install_gpu_local = function(local_path, yes, lib_dir) "sentinel"
    )
    expect_identical(install_gpu(yes = TRUE, local_path = src), "sentinel")
})
