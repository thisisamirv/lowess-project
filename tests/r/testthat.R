# extendr S3 registration requires NOT_CRAN=true.
# Set it here so covr/pkgcheck always measure coverage correctly.
Sys.setenv(NOT_CRAN = "true")

library(testthat)
library(rfastlowess)

test_check("rfastlowess")
