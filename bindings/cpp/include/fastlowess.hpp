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
#include <cstdint>
#include <cstring>
#include <limits>
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
constexpr int k_default_overlap = 500;
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
  static Expected make_error(std::string msg) {
    return Expected(std::move(msg), ErrorTag{});
  }

  bool has_value() const { return has_val_; }

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
  bool parallel = true;

  // Cross-validation options
  std::vector<double> cv_fractions;
  std::string cv_method = "kfold";
  int cv_k = detail::k_default_cv_k;
  /// Seed for cross-validation RNG (0 = unset / random).
  uint64_t cv_seed = 0;

  /// Per-observation case weights. When non-empty, must have the same length
  /// as the data passed to fit(). All values must be finite and non-negative.
  std::vector<double> custom_weights;
};

/**
 * @brief Options for streaming LOWESS.
 */
struct StreamingOptions : public LowessOptions {
  int chunk_size = detail::k_default_chunk_size;
  int overlap = detail::k_default_overlap;
  std::string merge_strategy =
      "weighted_average"; ///< weighted_average, average, take_first, take_last
};

/**
 * @brief Options for online LOWESS.
 *
 * Mirrors LowessOptions fields but defaults parallel to false, since
 * online LOWESS processes one point at a time and parallelism rarely helps.
 */
struct OnlineOptions {
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
  bool parallel = false; ///< Parallelism rarely helps for single-point updates

  std::vector<double> custom_weights;

  int window_capacity = detail::k_default_window_capacity;
  int min_points = 3;
  std::string update_mode = "full";
};

/**
 * @brief Result of a single online update step.
 *
 * Call has_value() to check if the window is ready.  When false, the window
 * is still filling and no smoothed estimate is available yet.  All optional
 * fields are NaN when not computed.
 */
class OnlineOutput {
public:
  /// True when the window has enough points to produce a smoothed estimate.
  bool has_value() const { return has_value_; }

  /// Smoothed value for the latest point (valid only when has_value() == true).
  double smoothed() const { return smoothed_; }

  /// Standard error (NaN if not computed).
  double std_error() const { return std_error_; }

  /// Residual y − smoothed (NaN if not computed).
  double residual() const { return residual_; }

  /// Robustness weight for the latest point (NaN if not computed).
  double robustness_weight() const { return robustness_weight_; }

  /// Number of robustness iterations performed (−1 if not applicable).
  int iterations_used() const { return iterations_used_; }

private:
  friend class OnlineLowess;
  template <typename U> friend class Expected;
  OnlineOutput() =
      default; ///< Constructs an empty (has_value==false) instance.
  explicit OnlineOutput(const fastlowess_CppOnlineOutput &raw)
      : has_value_(raw.has_value != 0), smoothed_(raw.smoothed),
        std_error_(raw.std_error), residual_(raw.residual),
        robustness_weight_(raw.robustness_weight),
        iterations_used_(raw.iterations_used) {}

  bool has_value_ = false;
  double smoothed_ = 0.0;
  double std_error_ = std::numeric_limits<double>::quiet_NaN();
  double residual_ = std::numeric_limits<double>::quiet_NaN();
  double robustness_weight_ = std::numeric_limits<double>::quiet_NaN();
  int iterations_used_ = -1;
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

  bool has_value() const { return !std::isnan(rmse_); }

  double rmse() const { return rmse_; }
  double mae() const { return mae_; }
  double r_squared() const { return r_squared_; }
  double aic() const { return aic_; }
  double aicc() const { return aicc_; }
  double effective_df() const { return effective_df_; }
  double residual_sd() const { return residual_sd_; }

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
  double x_value(size_t index) const { return result_.x[index]; }

  /// Access smoothed y value at index
  double y_value(size_t index) const { return result_.y[index]; }

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
    if (result_.residuals != nullptr) {
      return std::vector<double>(result_.residuals,
                                 result_.residuals + result_.n);
    }
    return {};
  }

  /// Get standard errors (empty if not computed)
  std::vector<double> standard_errors() const {
    if (result_.standard_errors != nullptr) {
      return std::vector<double>(result_.standard_errors,
                                 result_.standard_errors + result_.n);
    }
    return {};
  }

  /// Get confidence interval lower bounds
  std::vector<double> confidence_lower() const {
    if (result_.confidence_lower != nullptr) {
      return std::vector<double>(result_.confidence_lower,
                                 result_.confidence_lower + result_.n);
    }
    return {};
  }

  /// Get confidence interval upper bounds
  std::vector<double> confidence_upper() const {
    if (result_.confidence_upper != nullptr) {
      return std::vector<double>(result_.confidence_upper,
                                 result_.confidence_upper + result_.n);
    }
    return {};
  }

  /// Get prediction interval lower bounds
  std::vector<double> prediction_lower() const {
    if (result_.prediction_lower != nullptr) {
      return std::vector<double>(result_.prediction_lower,
                                 result_.prediction_lower + result_.n);
    }
    return {};
  }

  /// Get prediction interval upper bounds
  std::vector<double> prediction_upper() const {
    if (result_.prediction_upper != nullptr) {
      return std::vector<double>(result_.prediction_upper,
                                 result_.prediction_upper + result_.n);
    }
    return {};
  }

  /// Get robustness weights (empty if not computed)
  std::vector<double> robustness_weights() const {
    if (result_.robustness_weights != nullptr) {
      return std::vector<double>(result_.robustness_weights,
                                 result_.robustness_weights + result_.n);
    }
    return {};
  }

  /// Get cross-validation scores (empty if not computed)
  std::vector<double> cv_scores() const {
    if (result_.cv_scores != nullptr) {
      return std::vector<double>(result_.cv_scores,
                                 result_.cv_scores + result_.cv_scores_len);
    }
    return {};
  }

  /// Fraction used for smoothing
  double fraction_used() const { return result_.fraction_used; }

  /// Number of iterations performed (-1 if not available)
  int iterations_used() const { return result_.iterations_used; }

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
    if (options.cv_seed > 0) {
      cpp_lowess_set_cv_seed(ptr_, static_cast<unsigned long>(options.cv_seed));
    }
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
                             const std::vector<double> &y_values,
                             const std::vector<double> &custom_weights = {}) {
    if (x_values.size() != y_values.size()) {
      return Expected<LowessResult>::make_error(
          "x and y must have the same length");
    }
    if (x_values.empty()) {
      return Expected<LowessResult>::make_error(
          "Input arrays must not be empty");
    }

    auto result =
        cpp_lowess_fit(ptr_, x_values.data(), y_values.data(),
                       static_cast<unsigned long>(x_values.size()),
                       custom_weights.empty() ? nullptr : custom_weights.data(),
                       static_cast<unsigned long>(custom_weights.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::make_error(error_msg);
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
        options.merge_strategy.c_str(), options.confidence_intervals,
        options.prediction_intervals);
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

  Expected<LowessResult> process_chunk(const std::vector<double> &x_values,
                                       const std::vector<double> &y_values) {
    if (expect_finalized_) {
      return Expected<LowessResult>::make_error("Model already finalized");
    }
    if (x_values.size() != y_values.size()) {
      return Expected<LowessResult>::make_error("x and y length mismatch");
    }

    auto result =
        cpp_streaming_process(ptr_, x_values.data(), y_values.data(),
                              static_cast<unsigned long>(x_values.size()));

    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::make_error(error_msg);
    }
    return Expected<LowessResult>(LowessResult(result));
  }

  Expected<LowessResult> finalize() {
    if (expect_finalized_) {
      return Expected<LowessResult>::make_error("Model already finalized");
    }
    expect_finalized_ = true;

    auto result = cpp_streaming_finalize(ptr_);
    if (result.error != nullptr) {
      const std::string error_msg(result.error);
      cpp_lowess_free_result(&result);
      return Expected<LowessResult>::make_error(error_msg);
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
        options.return_diagnostics ? 1 : 0, options.return_residuals ? 1 : 0,
        options.zero_weight_fallback.c_str(), options.auto_converge,
        options.parallel ? 1 : 0, options.window_capacity, options.min_points,
        options.update_mode.c_str(), options.confidence_intervals,
        options.prediction_intervals);
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

  /**
   * @brief Add a single point and return the smoothed value.
   *
   * Returns an OnlineOutput. Call has_value() to check if the window is ready.
   * When false, the window is still filling.
   */
  Expected<OnlineOutput> add_point(double x, double y) {
    auto raw = cpp_online_add_point(ptr_, x, y);

    if (raw.error != nullptr) {
      const std::string error_msg(raw.error);
      cpp_online_free_output(&raw);
      return Expected<OnlineOutput>::make_error(error_msg);
    }
    return Expected<OnlineOutput>(OnlineOutput(raw));
  }

private:
  fastlowess_CppOnlineLowess *ptr_ = nullptr;
};

} // namespace fastlowess

#endif // FASTLOWESS_HPP
