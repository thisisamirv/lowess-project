#' @srrstats {G5.3} No NA/NaN in validated outputs.
#' @srrstats {G5.8, G5.8a} Edge condition tests for GPU backend helpers.
# Tests targeting uncovered lines in gpu.R:
#   gpu_available / gpu_enabled (extendr-wrappers.R:110)
#   check_gpu_backend (both branches)
#   gpu_asset_info (all three platform branches, via mocking)
#   gpu_confirm_download (both branches)
#   gpu_download_to (success and failure, via mocking)
#   gpu_lib_dir (windows multi-arch and default branches)
#   install_gpu (already-active and needs-confirmation branches)

check_gpu_backend <- getFromNamespace("check_gpu_backend", "rfastlowess")
gpu_asset_info <- getFromNamespace("gpu_asset_info", "rfastlowess")
gpu_confirm_download <- getFromNamespace("gpu_confirm_download", "rfastlowess")
gpu_download_to <- getFromNamespace("gpu_download_to", "rfastlowess")
gpu_lib_dir <- getFromNamespace("gpu_lib_dir", "rfastlowess")
read_line <- getFromNamespace("read_line", "rfastlowess")

# ── read_line ─────────────────────────────────────────────────────────────────

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
    info <- gpu_asset_info("1.0.0")
    expect_identical(info$ext, ".dll")
    expect_true(grepl("windows-x86_64\\.dll$", info$asset))
})

test_that("gpu_asset_info handles macOS arm64 platform", {
    testthat::local_mocked_bindings(
        `Sys.info` = function() c(sysname = "Darwin", machine = "arm64"),
        .package = "base"
    )
    info <- gpu_asset_info("1.0.0")
    expect_identical(info$ext, ".so")
    expect_true(grepl("macos-aarch64\\.so$", info$asset))
})

test_that("gpu_asset_info handles Linux platform", {
    testthat::local_mocked_bindings(
        `Sys.info` = function() c(sysname = "Linux", machine = "x86_64"),
        .package = "base"
    )
    info <- gpu_asset_info("1.0.0")
    expect_identical(info$ext, ".so")
    expect_true(grepl("linux-x86_64\\.so$", info$asset))
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
    gpu_download_to("https://example.com/lib.so", ".so", dest)
    expect_true(file.exists(dest))
    expect_gt(file.size(dest), 0)
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
        gpu_download_to = function(url, ext, dest) invisible(TRUE)
    )
    expect_message(
        result <- install_gpu(yes = TRUE),
        "GPU backend installed at"
    )
    expect_true(isTRUE(result))
})
