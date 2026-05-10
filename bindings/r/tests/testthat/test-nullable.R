#' @srrstats {G2.13, G2.14} Nullable handles NULL/non-NULL optional values.
test_that("Nullable returns NULL for NULL input", {
    expect_null(Nullable(NULL))
})

test_that("Nullable returns value for non-NULL input", {
    expect_identical(Nullable(5), 5)
    expect_identical(Nullable("hello"), "hello")
    expect_identical(Nullable(c(1, 2, 3)), c(1, 2, 3))
})
