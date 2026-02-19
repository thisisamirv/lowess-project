#' @srrstats {G2.13, G2.14} Nullable handles NULL/non-NULL optional values.
test_that("Nullable returns NULL for NULL input", {
    expect_null(Nullable(NULL))
})

test_that("Nullable returns value for non-NULL input", {
    expect_equal(Nullable(5), 5)
    expect_equal(Nullable("hello"), "hello")
    expect_equal(Nullable(c(1, 2, 3)), c(1, 2, 3))
})
