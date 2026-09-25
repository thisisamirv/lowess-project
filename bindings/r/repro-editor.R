library(rfastlowess)
cat("version", as.character(packageVersion("rfastlowess")), "\n")
run_one <- function(label, x, y, fraction, iterations) {
  cat("\n== ", label, " ==\n", sep = "")
  e <- stats::lowess(x, y, f = fraction, iter = iterations)
  m_default <- Lowess(
    fraction = fraction,
    iterations = iterations,
    boundary_policy = "noboundary",
    scaling_method = "mar",
    outputs = "sorted"
  )
  r_default <- fit(m_default, x, y)
  m_r <- Lowess(
    fraction = fraction,
    iterations = iterations,
    boundary_policy = "noboundary",
    scaling_method = "mar",
    zero_weight_fallback = "return_original",
    outputs = "sorted"
  )
  r_r <- fit(m_r, x, y)
  cat("default y identical to R? ", identical(r_default$y, e$y), "\n", sep = "")
  cat("return_original y identical to R? ", identical(r_r$y, e$y), "\n", sep = "")
  cat("max abs default: ", format(max(abs(r_default$y - e$y)), digits = 16), "\n", sep = "")
  cat("max abs return_original: ", format(max(abs(r_r$y - e$y)), digits = 16), "\n", sep = "")
  cat("default y: ", paste(format(r_default$y, digits = 16, scientific = FALSE), collapse = " "), "\n", sep = "")
  cat("R y:       ", paste(format(e$y, digits = 16, scientific = FALSE), collapse = " "), "\n", sep = "")
  cat("override y:", paste(format(r_r$y, digits = 16, scientific = FALSE), collapse = " "), "\n", sep = "")
}
run_one(
  "example 1",
  c(-1.931272, -1.688085, 0, 0, -3.542724),
  c(1.959219, -0.769967, 0, 0, 0),
  0.7808,
  116L
)
run_one(
  "example 2",
  c(0, -0.81768, 2.59103, 2.53722, -1.52132),
  c(0, 0, 1.6702, -1.6494, 1.2770),
  0.6313,
  173L
)
