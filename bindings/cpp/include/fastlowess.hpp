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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include the C header
#include "fastlowess.h"

namespace fastlowess {

namespace detail {
constexpr double k_default_fraction = 0.67;
constexpr int k_default_cv_k = 5;
constexpr int k_default_chunk_size = 5000;
constexpr int k_default_window_capacity = 1000;
} // namespace detail

/**
 * @brief Exception thrown when LOWESS operation fails.
 */
class LowessError : public std::runtime_error {
public:
  explicit LowessError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief A result type that holds either a value or an error.
 * Mimics std::expected (C++23) behavior.
 */
template <typename T> class Expected {
public:
  // Success constructor
  Expected(T val) : val_(std::move(val)), has_val_(true) {}

  // Error constructor
  struct ErrorTag {};
  static Expected makeError(std::string msg) {
    return Expected(std::move(msg), ErrorTag{});
  }

  bool hasValue() const { return has_val_; }

  explicit operator bool() const { return has_val_; }

  T &value() & {
    if (!has_val_) {
      throw LowessError(err_);
    }
    return val_;
  }

  const T &value() const & {
    if (!has_val_) {
      throw LowessError(err_);
    }
    return val_;
  }

  T &&value() && {
    if (!has_val_) {
      throw LowessError(err_);
    }
    return std::move(val_);
  }

  const std::string &error() const {
    if (has_val_) {
      throw LowessError("Bad expected access: has value");
    }
    return err_;
  }

private:
  Expected(std::string err, ErrorTag error_tag)
      : err_(std::move(err)), has_val_(false) {
    static_cast<void>(error_tag);
  }

  // We store both to avoid manual union management, relying on T's cheap
  // default ctor. LowessResult's default ctor is cheap (zero-init).
  T val_;
  std::string err_;
  bool has_val_;
};

/**
 * @brief Options for configuring LOWESS smoothing.
 */
struct LowessOptions {
  double fraction = detail::k_default_fraction; ///< Smoothing fraction (0, 1]
  int iterations = 3;                           ///< Robustness iterations
  double delta = NAN; ///< Interpolation threshold (NaN = auto)

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
  int cv_k = detail::k_default_cv_k;
};

/**
 * @brief Options for streaming LOWESS.
 */
struct StreamingOptions : public LowessOptions {
  int chunk_size = detail::k_default_chunk_size;
  int overlap = -1;                        ///< -1 for auto
  std::string merge_strategy = "weighted"; ///< average, weighted, first, last
};

/**
 * @brief Options for online LOWESS.
 */
struct OnlineOptions : public LowessOptions {
  int window_capacity = detail::k_default_window_capacity;
  int min_points = 2;
  std::string update_mode = "full";
};

/**
 * @brief Diagnostics from LOWESS fitting.
 */
class Diagnostics {
public:
  Diagnostics() = default;

  explicit Diagnostics(const fastlowess_CppLowessResult &result)
      : rmse_(result.rmse), mae_(result.mae), r_squared_(result.r_squared),
        aic_(result.aic), aicc_(result.aicc),
        effective_df_(result.effective_df), residual_sd_(result.residual_sd) {}

  bool hasValue() const { return !std::isnan(rmse_); }

  double rmse() const { return rmse_; }
  double mae() const { return mae_; }
  double rSquared() const { return r_squared_; }
  double aic() const { return aic_; }
  double aicc() const { return aicc_; }
  double effectiveDf() const { return effective_df_; }
  double residualSd() const { return residual_sd_; }

private:
  double rmse_ = NAN;
  double mae_ = NAN;
  double r_squared_ = NAN;
  double aic_ = NAN;
  double aicc_ = NAN;
  double effective_df_ = NAN;
  double residual_sd_ = NAN;
};

/**
 * @brief Result of LOWESS smoothing operation.
 *
 * RAII wrapper that automatically frees the underlying C result.
 */
class LowessResult {
public:
  LowessResult() = default;

  explicit LowessResult(const fastlowess_CppLowessResult &c_result)
      : result_(c_result) {}

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
    return result_.error != nullptr ? std::string(result_.error) : "";
  }

  /// Access x value at index
  double xValue(size_t index) const { return result_.x[index]; }

  /// Access smoothed y value at index
  double yValue(size_t index) const { return result_.y[index]; }

  /// Get x values as vector
  std::vector<double> xVector() const {
    return std::vector<double>(result_.x, result_.x + result_.n);
  }

  /// Get smoothed y values as vector
  std::vector<double> yVector() const {
    return std::vector<double>(result_.y, result_.y + result_.n);
  }

  /// Get residuals (empty if not computed)
  std::vector<double> residuals() const {
    if (result_.residuals != nullptr) {
      return std::vector<double>(result_.residuals,
                                 result_.residuals + result_.n);
    }
    return {};
  }

  /// Get standard errors (empty if not computed)
  std::vector<double> standardErrors() const {
    if (result_.standard_errors != nullptr) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidenceLower() const {
    if (result_.confidence_lower != nullptr) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidenceUpper() const {
    if (result_.confidence_upper != nullptr) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> predictionLower() const {
    if (result_.prediction_lower != nullptr) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> predictionUpper() const {
    if (result_.prediction_upper != nullptr) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustnessWeights() const {
    if (result_.robustness_weights != nullptr) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fractionUsed() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterationsUsed() const { return result_.iterations_used; }

  /// Get diagnostics
  Diagnostics diagnostics() const { return Diagnostics(result_); }

private:
  fastlowess_CppLowessResult result_ = {};
};

/**
 * @brief Batch LOWESS model.
 */
class Lowess {
public:
  explicit Lowess(const LowessOptions &options = {}) {
    ptr_ = cpp_lowess_new(
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.confidence_intervals, options.prediction_intervals,
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.cv_fractions.empty() ? nullptr : options.cv_fractions.data(),
        static_cast<unsigned long>(options.cv_fractions.size()),
        options.cv_method.c_str(), options.cv_k, options.parallel ? 1 : 0);
  }

  ~Lowess() {
    if (ptr_ != nullptr) {
      cpp_lowess_free(ptr_);
    }
  }

  // Non-copyable
  Lowess(const Lowess &) = delete;
  Lowess &operator=(const Lowess &) = delete;

  // Move-able
  Lowess(Lowess &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  Lowess &operator=(Lowess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_lowess_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LowessResult> fit(const std::vector<double> &x_values,
                             const std::vector<double> &y_values) {
    if (x_values.size() != y_values.size()) {
      return Expected<LowessResult>::makeError(
          "x and y must have the same length");
    }
    if (x_values.empty()) {
      return Expected<LowessResult>::makeError(
          "Input arrays must not be empty");
    }

    auto result = cpp_lowess_fit(ptr_, x_values.data(), y_values.data(),
                                 static_cast<unsigned long>(x_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::makeError(error_msg);
    }

    return Expected<LowessResult>(LowessResult(result));
  }

private:
  fastlowess_CppLowess *ptr_ = nullptr;
};

/**
 * @brief Streaming LOWESS model.
 */
class StreamingLowess {
public:
  explicit StreamingLowess(const StreamingOptions &options = {}) {
    ptr_ = cpp_streaming_new(
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.chunk_size, options.overlap,
        options.merge_strategy.c_str());
  }

  ~StreamingLowess() {
    if (ptr_ != nullptr) {
      cpp_streaming_free(ptr_);
    }
  }

  StreamingLowess(const StreamingLowess &) = delete;
  StreamingLowess &operator=(const StreamingLowess &) = delete;
  StreamingLowess(StreamingLowess &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  StreamingLowess &operator=(StreamingLowess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_streaming_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LowessResult> processChunk(const std::vector<double> &x_values,
                                      const std::vector<double> &y_values) {
    if (expect_finalized_) {
      return Expected<LowessResult>::makeError("Model already finalized");
    }
    if (x_values.size() != y_values.size()) {
      return Expected<LowessResult>::makeError("x and y length mismatch");
    }

    auto result =
        cpp_streaming_process(ptr_, x_values.data(), y_values.data(),
                              static_cast<unsigned long>(x_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::makeError(error_msg);
    }
    return Expected<LowessResult>(LowessResult(result));
  }

  Expected<LowessResult> finalize() {
    if (expect_finalized_) {
      return Expected<LowessResult>::makeError("Model already finalized");
    }
    expect_finalized_ = true;

    auto result = cpp_streaming_finalize(ptr_);
    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::makeError(error_msg);
    }
    return Expected<LowessResult>(LowessResult(result));
  }

private:
  fastlowess_CppStreamingLowess *ptr_ = nullptr;
  bool expect_finalized_ = false;
};

/**
 * @brief Online LOWESS model.
 */
class OnlineLowess {
public:
  explicit OnlineLowess(const OnlineOptions &options = {}) {
    ptr_ = cpp_online_new(
        options.fraction, options.iterations, options.delta,
        options.weight_function.c_str(), options.robustness_method.c_str(),
        options.scaling_method.c_str(), options.boundary_policy.c_str(),
        options.return_robustness_weights ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.window_capacity, options.min_points,
        options.update_mode.c_str());
  }

  ~OnlineLowess() {
    if (ptr_ != nullptr) {
      cpp_online_free(ptr_);
    }
  }

  OnlineLowess(const OnlineLowess &) = delete;
  OnlineLowess &operator=(const OnlineLowess &) = delete;
  OnlineLowess(OnlineLowess &&other) noexcept : ptr_(other.ptr_) {
    other.ptr_ = nullptr;
  }
  OnlineLowess &operator=(OnlineLowess &&other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cpp_online_free(ptr_);
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  Expected<LowessResult> addPoints(const std::vector<double> &x_values,
                                   const std::vector<double> &y_values) {
    if (x_values.size() != y_values.size()) {
      return Expected<LowessResult>::makeError("x and y length mismatch");
    }

    auto result =
        cpp_online_add_points(ptr_, x_values.data(), y_values.data(),
                              static_cast<unsigned long>(x_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::makeError(error_msg);
    }
    return Expected<LowessResult>(LowessResult(result));
  }

private:
  fastlowess_CppOnlineLowess *ptr_ = nullptr;
};

} // namespace fastlowess

#endif // FASTLOWESS_HPP
