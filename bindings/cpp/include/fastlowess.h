#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// Result struct that can be passed across FFI boundary.
/// All arrays are allocated by Rust and must be freed by Rust.
struct fastlowess_CppLowessResult {
  /// Sorted x values (length = n)
  double *x;
  /// Smoothed y values (length = n)
  double *y;
  /// Number of data points
  unsigned long n;
  /// Standard errors (NULL if not computed)
  double *standard_errors;
  /// Lower confidence bounds (NULL if not computed)
  double *confidence_lower;
  /// Upper confidence bounds (NULL if not computed)
  double *confidence_upper;
  /// Lower prediction bounds (NULL if not computed)
  double *prediction_lower;
  /// Upper prediction bounds (NULL if not computed)
  double *prediction_upper;
  /// Residuals (NULL if not computed)
  double *residuals;
  /// Robustness weights (NULL if not computed)
  double *robustness_weights;
  /// Fraction used for smoothing
  double fraction_used;
  /// Number of iterations performed (-1 if not available)
  int iterations_used;
  /// Diagnostics (NaN if not computed)
  double rmse;
  double mae;
  double r_squared;
  double aic;
  double aicc;
  double effective_df;
  double residual_sd;
  /// Error message (NULL if no error)
  char *error;
};

extern "C" {

/// LOWESS smoothing with batch adapter.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
fastlowess_CppLowessResult cpp_lowess_smooth(const double *x,
                                             const double *y,
                                             unsigned long n,
                                             double fraction,
                                             int iterations,
                                             double delta,
                                             const char *weight_function,
                                             const char *robustness_method,
                                             const char *scaling_method,
                                             const char *boundary_policy,
                                             double confidence_intervals,
                                             double prediction_intervals,
                                             int return_diagnostics,
                                             int return_residuals,
                                             int return_robustness_weights,
                                             const char *zero_weight_fallback,
                                             double auto_converge,
                                             const double *cv_fractions,
                                             unsigned long cv_fractions_len,
                                             const char *cv_method,
                                             int cv_k,
                                             int parallel);

/// Streaming LOWESS for large datasets.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
fastlowess_CppLowessResult cpp_lowess_streaming(const double *x,
                                                const double *y,
                                                unsigned long n,
                                                double fraction,
                                                int chunk_size,
                                                int overlap,
                                                int iterations,
                                                double delta,
                                                const char *weight_function,
                                                const char *robustness_method,
                                                const char *scaling_method,
                                                const char *boundary_policy,
                                                double auto_converge,
                                                int return_diagnostics,
                                                int return_residuals,
                                                int return_robustness_weights,
                                                const char *zero_weight_fallback,
                                                const char *merge_strategy,
                                                int parallel);

/// Online LOWESS with sliding window.
///
/// # Safety
/// All pointer arguments must be valid and point to arrays of the specified length.
fastlowess_CppLowessResult cpp_lowess_online(const double *x,
                                             const double *y,
                                             unsigned long n,
                                             double fraction,
                                             int window_capacity,
                                             int min_points,
                                             int iterations,
                                             double delta,
                                             const char *weight_function,
                                             const char *robustness_method,
                                             const char *scaling_method,
                                             const char *boundary_policy,
                                             const char *update_mode,
                                             double auto_converge,
                                             int return_robustness_weights,
                                             const char *zero_weight_fallback,
                                             int parallel);

/// Free a CppLowessResult allocated by Rust.
///
/// # Safety
/// The result pointer must have been returned by one of the cpp_lowess_* functions.
void cpp_lowess_free_result(fastlowess_CppLowessResult *result);

}  // extern "C"
