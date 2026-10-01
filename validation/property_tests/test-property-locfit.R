#' @srrstats {G5.4b, G5.10} Randomized comparison against Locfit's independent
#'   local-polynomial regression implementation.
#' @noRd

test_that("matches Locfit for shared local-linear options", {
    suppressPackageStartupMessages(library(locfit))

    had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (had_seed) {
        old_seed <- get(".Random.seed", envir = .GlobalEnv)
    }
    on.exit({
        if (had_seed) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (exists(".Random.seed", envir = .GlobalEnv)) {
            rm(".Random.seed", envir = .GlobalEnv)
        }
    })
    set.seed(20261001)

    kernel_names <- c(
        "tricube", "epanechnikov", "biweight", "triangle", "gaussian"
    )
    locfit_kernels <- c("tcub", "epan", "bisq", "tria", "gauss")

    property <- function(xy, fraction, kernel_index) {
        if (length(xy[[1]]) < 8L) {
            return(expect_true(TRUE))
        }
        ord <- order(xy[[1]])
        gaps <- 1 + abs(diff(c(0, xy[[1]][ord])))
        x <- cumsum(gaps)
        x <- 2 * x / max(x) - 1
        y <- as.double(xy[[2]][ord])
        case_weights <- 0.1 + abs(xy[[1]][ord]) / 100
        kernel <- kernel_names[[kernel_index]]
        locfit_kernel <- locfit_kernels[[kernel_index]]

        result <- fit(
            Lowess(
                fraction = fraction,
                iterations = 0L,
                delta = 0,
                weight_function = kernel,
                boundary_policy = "noboundary"
            ),
            x,
            y,
            custom_weights = case_weights
        )
        reference <- locfit::smooth.lf(
            x,
            y,
            xev = x,
            direct = TRUE,
            alpha = fraction,
            deg = 1,
            kern = locfit_kernel,
            weights = case_weights
        )

        expect_equal(result$y, as.double(reference$y), tolerance = 1e-10)
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(8L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(8L, 40L))
        ),
        fraction = quickcheck::double_bounded(0.2, 0.8, len = 1L),
        kernel_index = quickcheck::integer_bounded(1L, 5L, len = 1L),
        property = property,
        tests = 100L,
        discards = 500L
    )

    robust_property <- function(xy) {
        if (length(xy[[1]]) < 20L) {
            return(expect_true(TRUE))
        }
        ord <- order(xy[[1]])
        gaps <- 1 + abs(diff(c(0, xy[[1]][ord])))
        x <- cumsum(gaps)
        x <- 2 * x / max(x) - 1
        y <- as.double(xy[[2]][ord]) + sin(pi * x) + 0.2 * sin(13 * x)
        spike <- ceiling(length(y) / 2)
        y[[spike]] <- y[[spike]] + 4

        result <- fit(
            Lowess(
                fraction = 0.4,
                iterations = 3L,
                delta = 0,
                weight_function = "tricube",
                scaling_method = "mar",
                boundary_policy = "noboundary",
                zero_weight_fallback = "return_original"
            ),
            x,
            y
        )
        # Locfit robust uses three passes by default; an explicit `iter`
        # argument is forwarded to locfit.raw and rejected by Locfit 1.5.
        reference <- locfit::locfit.robust(
            x,
            y,
            weights = rep(1, length(x)),
            alpha = 0.4,
            deg = 1,
            kern = "tcub",
            ev = locfit::dat()
        )

        expect_equal(
            result$y,
            as.double(stats::predict(reference, where = "fitp")),
            tolerance = 1e-10
        )
    }

    quickcheck::for_all(
        xy = quickcheck::equal_length(
            quickcheck::double_bounded(-100, 100, len = c(20L, 40L)),
            quickcheck::double_bounded(-100, 100, len = c(20L, 40L))
        ),
        property = robust_property,
        tests = 100L,
        discards = 500L
    )
})
