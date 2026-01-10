/**
 * @file fastlowess.hpp
 * @brief C++ wrapper for fastLowess library
 *
 * Provides idiomatic C++ access to LOWESS smoothing with RAII,
 * exceptions, and STL container support.
 */

#ifndef FASTLOWESS_HPP
#define FASTLOWESS_HPP

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Include the C header
extern "C" {
#include "fastlowess.h"
}

namespace fastlowess {

/**
 * @brief Exception thrown when LOWESS operation fails.
 */
class LowessError : public std::runtime_error {
public:
  explicit LowessError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief Options for configuring LOWESS smoothing.
 */
struct LowessOptions {
  double fraction = 0.67; ///< Smoothing fraction (0, 1]
  int iterations = 3;     ///< Robustness iterations
  double delta = NAN;     ///< Interpolation threshold (NaN = auto)

  std::string weight_function = "tricube";
  std::string robustness_method = "bisquare";
  std::string scaling_method = "mad";
  std::string boundary_policy = "extend";
  std::string zero_weight_fallback = "use_local_mean";

  double confidence_intervals = NAN; ///< Confidence level (NaN = disabled)
  double prediction_intervals = NAN; ///< Prediction level (NaN = disabled)
  double auto_converge = NAN;        ///< Auto-convergence threshold

  bool return_diagnostics = false;
  bool return_residuals = false;
  bool return_robustness_weights = false;
  bool parallel = false;

  // Cross-validation options
  std::vector<double> cv_fractions;
  std::string cv_method = "kfold";
  int cv_k = 5;
};

/**
 * @brief Options for streaming LOWESS.
 */
struct StreamingOptions : public LowessOptions {
  int chunk_size = 5000;
  int overlap = -1; ///< -1 for auto
  std::string merge_strategy = "weighted"; ///< average, weighted, first, last
};

/**
 * @brief Options for online LOWESS.
 */
struct OnlineOptions : public LowessOptions {
  int window_capacity = 1000;
  int min_points = 2;
  std::string update_mode = "full";
};

/**
 * @brief Diagnostics from LOWESS fitting.
 */
struct Diagnostics {
  double rmse = NAN;
  double mae = NAN;
  double r_squared = NAN;
  double aic = NAN;
  double aicc = NAN;
  double effective_df = NAN;
  double residual_sd = NAN;

  bool has_value() const { return !std::isnan(rmse); }
};

/**
 * @brief Result of LOWESS smoothing operation.
 *
 * RAII wrapper that automatically frees the underlying C result.
 */
class LowessResult {
public:
  LowessResult() = default;

  explicit LowessResult(fastlowess_CppLowessResult &&c_result)
      : result_(std::move(c_result)) {
    c_result = fastlowess_CppLowessResult{}; // Clear moved-from result
  }

  ~LowessResult() {
    if (result_.n > 0) {
      cpp_lowess_free_result(&result_);
    }
  }

  // Move-only
  LowessResult(const LowessResult &) = delete;
  LowessResult &operator=(const LowessResult &) = delete;

  LowessResult(LowessResult &&other) noexcept : result_(other.result_) {
    other.result_ = fastlowess_CppLowessResult{};
  }

  LowessResult &operator=(LowessResult &&other) noexcept {
    if (this != &other) {
      if (result_.n > 0) {
        cpp_lowess_free_result(&result_);
      }
      result_ = other.result_;
      other.result_ = fastlowess_CppLowessResult{};
    }
    return *this;
  }

  /// Number of data points
  size_t size() const { return static_cast<size_t>(result_.n); }

  /// Check if result is valid
  bool valid() const { return result_.n > 0 && result_.error == nullptr; }

  /// Get error message (empty if no error)
  std::string error() const {
    return result_.error ? std::string(result_.error) : "";
  }

  /// Access x value at index
  double x(size_t i) const { return result_.x[i]; }

  /// Access smoothed y value at index
  double y(size_t i) const { return result_.y[i]; }

  /// Get x values as vector
  std::vector<double> x_vector() const {
    return std::vector<double>(result_.x, result_.x + result_.n);
  }

  /// Get smoothed y values as vector
  std::vector<double> y_vector() const {
    return std::vector<double>(result_.y, result_.y + result_.n);
  }

  /// Get residuals (empty if not computed)
  std::vector<double> residuals() const {
    if (result_.residuals) {
      return std::vector<double>(result_.residuals,
                                 result_.residuals + result_.n);
    }
    return {};
  }

  /// Get standard errors (empty if not computed)
  std::vector<double> standard_errors() const {
    if (result_.standard_errors) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidence_lower() const {
    if (result_.confidence_lower) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidence_upper() const {
    if (result_.confidence_upper) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> prediction_lower() const {
    if (result_.prediction_lower) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> prediction_upper() const {
    if (result_.prediction_upper) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustness_weights() const {
    if (result_.robustness_weights) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fraction_used() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterations_used() const { return result_.iterations_used; }

  /// Get diagnostics
  Diagnostics diagnostics() const {
    return Diagnostics{result_.rmse,       result_.mae,  result_.r_squared,
                       result_.aic,        result_.aicc, result_.effective_df,
                       result_.residual_sd};
  }

private:
  fastlowess_CppLowessResult result_ = {};
};

/**
 * @brief Perform batch LOWESS smoothing.
 *
 * @param x Independent variable values
 * @param y Dependent variable values
 * @param options Smoothing options
 * @return LowessResult containing smoothed values
 * @throws LowessError if smoothing fails
 */
inline LowessResult smooth(const std::vector<double> &x,
                           const std::vector<double> &y,
                           const LowessOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LowessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LowessError("Input arrays must not be empty");
  }

  fastlowess_CppLowessResult result = cpp_lowess_smooth(
      x.data(), y.data(), x.size(), options.fraction, options.iterations,
      options.delta, options.weight_function.c_str(),
      options.robustness_method.c_str(), options.scaling_method.c_str(),
      options.boundary_policy.c_str(), options.confidence_intervals,
      options.prediction_intervals, options.return_diagnostics ? 1 : 0,
      options.return_residuals ? 1 : 0,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.auto_converge,
      options.cv_fractions.empty() ? nullptr : options.cv_fractions.data(),
      options.cv_fractions.size(), options.cv_method.c_str(), options.cv_k,
      options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_lowess_free_result(&result);
    throw LowessError(error_msg);
  }

  return LowessResult(std::move(result));
}

/**
 * @brief Perform streaming LOWESS for large datasets.
 */
inline LowessResult streaming(const std::vector<double> &x,
                              const std::vector<double> &y,
                              const StreamingOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LowessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LowessError("Input arrays must not be empty");
  }

  fastlowess_CppLowessResult result = cpp_lowess_streaming(
      x.data(), y.data(), x.size(), options.fraction, options.chunk_size,
      options.overlap, options.iterations, options.delta,
      options.weight_function.c_str(), options.robustness_method.c_str(),
      options.scaling_method.c_str(), options.boundary_policy.c_str(),
      options.auto_converge, options.return_diagnostics ? 1 : 0,
      options.return_residuals ? 1 : 0,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.merge_strategy.c_str(),
      options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_lowess_free_result(&result);
    throw LowessError(error_msg);
  }

  return LowessResult(std::move(result));
}

/**
 * @brief Perform online LOWESS with sliding window.
 */
inline LowessResult online(const std::vector<double> &x,
                           const std::vector<double> &y,
                           const OnlineOptions &options = {}) {
  if (x.size() != y.size()) {
    throw LowessError("x and y must have the same length");
  }
  if (x.empty()) {
    throw LowessError("Input arrays must not be empty");
  }

  fastlowess_CppLowessResult result = cpp_lowess_online(
      x.data(), y.data(), x.size(), options.fraction, options.window_capacity,
      options.min_points, options.iterations, options.delta,
      options.weight_function.c_str(), options.robustness_method.c_str(),
      options.scaling_method.c_str(), options.boundary_policy.c_str(),
      options.update_mode.c_str(), options.auto_converge,
      options.return_robustness_weights ? 1 : 0,
      options.zero_weight_fallback.c_str(), options.parallel ? 1 : 0);

  if (result.error != nullptr) {
    std::string error_msg(result.error);
    cpp_lowess_free_result(&result);
    throw LowessError(error_msg);
  }

  return LowessResult(std::move(result));
}

} // namespace fastlowess

#endif // FASTLOWESS_HPP
