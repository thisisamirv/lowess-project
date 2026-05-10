/**
 * @file batch_smoothing.cpp
 * @brief fastlowess Batch Smoothing Example
 *
 * This example demonstrates batch LOWESS smoothing features:
 * - Basic smoothing with different parameters
 * - Robustness iterations for outlier handling
 * - Confidence and prediction intervals
 * - Diagnostics and cross-validation
 *
 * The Lowess class is the primary interface for
 * processing complete datasets that fit in memory.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../../bindings/cpp/include/fastlowess.hpp"

namespace {

constexpr size_t k_default_point_count = 100;
constexpr unsigned int k_random_seed = 42;
constexpr double k_noise_std_dev = 1.5;
constexpr double k_outlier_magnitude_min = 10.0;
constexpr double k_outlier_magnitude_max = 20.0;
constexpr double k_x_range_max = 50.0;
constexpr double k_trend_slope = 0.5;
constexpr double k_seasonal_amplitude = 5.0;
constexpr double k_seasonal_frequency = 0.5;
constexpr size_t k_outlier_divisor = 10;
constexpr double k_basic_fraction = 0.05;
constexpr double k_confidence_level = 0.95;
constexpr double k_linear_range_max = 10.0;
constexpr size_t k_linear_point_count = 50;
constexpr double k_linear_slope = 2.0;
constexpr double k_linear_intercept = 1.0;
constexpr double k_boundary_fraction = 0.6;

struct Data {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> y_true;
};

Data generateSampleData(size_t point_count = k_default_point_count) {
  Data data;
  data.x.resize(point_count);
  data.y.resize(point_count);
  data.y_true.resize(point_count);

  std::seed_seq generator_seed = {k_random_seed, k_random_seed, k_random_seed,
                                  k_random_seed};
  std::mt19937 generator(generator_seed);
  std::normal_distribution<> noise(0.0, k_noise_std_dev);
  std::uniform_real_distribution<> outlier_magnitude(k_outlier_magnitude_min,
                                                     k_outlier_magnitude_max);
  std::uniform_int_distribution<> outlier_sign(0, 1);

  for (size_t point_index = 0; point_index < point_count; ++point_index) {
    data.x[point_index] = static_cast<double>(point_index) * k_x_range_max /
                          static_cast<double>(point_count - 1);

    data.y_true[point_index] =
        (k_trend_slope * data.x[point_index]) +
        (k_seasonal_amplitude *
         std::sin(data.x[point_index] * k_seasonal_frequency));

    data.y[point_index] = data.y_true[point_index] + noise(generator);
  }

  const size_t outlier_count = point_count / k_outlier_divisor;
  std::uniform_int_distribution<size_t> outlier_index(0, point_count - 1);

  for (size_t outlier_number = 0; outlier_number < outlier_count;
       ++outlier_number) {
    const size_t point_index = outlier_index(generator);
    double outlier_value = outlier_magnitude(generator);
    if (outlier_sign(generator) == 0) {
      outlier_value = -outlier_value;
    }
    data.y[point_index] += outlier_value;
  }

  return data;
}

} // namespace

int main() {
  try {
    std::cout << "=== fastlowess Batch Smoothing Example ===\n";

    // 1. Generate Data
    auto data = generateSampleData(k_default_point_count);
    std::cout << "Generated " << data.x.size() << " data points with outliers."
              << '\n';

    // 2. Basic Smoothing (Default parameters)
    std::cout << "Running basic smoothing...\n";
    fastlowess::LowessOptions basic_opts;
    basic_opts.fraction = k_basic_fraction;
    basic_opts.iterations = 0;
    fastlowess::Lowess model_basic(basic_opts);
    auto res_basic = model_basic.fit(data.x, data.y).value();

    // 3. Robust Smoothing (IRLS)
    std::cout << "Running robust smoothing (3 iterations)...\n";
    fastlowess::LowessOptions robust_opts;
    robust_opts.fraction = k_basic_fraction;
    robust_opts.iterations = 3;
    robust_opts.robustness_method = "bisquare";
    robust_opts.return_robustness_weights = true;

    fastlowess::Lowess model_robust(robust_opts);
    auto res_robust = model_robust.fit(data.x, data.y).value();

    // 4. Uncertainty Quantification
    std::cout << "Computing confidence and prediction intervals..." << '\n';
    fastlowess::LowessOptions interval_opts;
    interval_opts.fraction = k_basic_fraction;
    interval_opts.confidence_intervals = k_confidence_level;
    interval_opts.prediction_intervals = k_confidence_level;
    interval_opts.return_diagnostics = true;

    fastlowess::Lowess model_intervals(interval_opts);
    auto res_intervals = model_intervals.fit(data.x, data.y).value();

    // 5. Cross-Validation for optimal fraction
    std::cout << "Running cross-validation to find optimal fraction..." << '\n';

    // Manual CV search
    const std::vector<double> fractions = {k_basic_fraction, 0.1, 0.2, 0.4};
    double best_fraction = 0.0;
    double min_rmse = std::numeric_limits<double>::max();

    for (const double fraction : fractions) {
      fastlowess::LowessOptions cv_opts;
      cv_opts.fraction = fraction;
      cv_opts.return_diagnostics = true;
      fastlowess::Lowess model(cv_opts);
      auto res_exp = model.fit(data.x, data.y);

      // Use non-throwing interface
      if (res_exp.hasValue()) {
        auto &res = res_exp.value();
        if (res.diagnostics().hasValue()) {
          const double rmse = res.diagnostics().rmse();
          if (rmse < min_rmse) {
            min_rmse = rmse;
            best_fraction = fraction;
          }
        }
      }
    }
    std::cout << "Optimal fraction found (manual CV): " << best_fraction
              << '\n';

    // Diagnostics Printout
    if (res_intervals.diagnostics().hasValue()) {
      const auto diag = res_intervals.diagnostics();
      std::cout << "\nFit Statistics (Intervals Model):\n";
      std::cout << " - R^2:   " << diag.rSquared() << '\n';
      std::cout << " - RMSE: " << diag.rmse() << '\n';
      std::cout << " - MAE:  " << diag.mae() << '\n';
    }

    // 6. Boundary Policy Comparison
    std::cout << "\nDemonstrating boundary policy effects on linear data..."
              << '\n';
    std::vector<double> linear_x(k_linear_point_count);
    std::vector<double> linear_y(k_linear_point_count);
    for (size_t point_index = 0; point_index < k_linear_point_count;
         ++point_index) {
      linear_x[point_index] = static_cast<double>(point_index) *
                              k_linear_range_max /
                              static_cast<double>(k_linear_point_count - 1);
      linear_y[point_index] =
          (k_linear_slope * linear_x[point_index]) + k_linear_intercept;
    }

    fastlowess::LowessOptions opt_ext;
    opt_ext.fraction = k_boundary_fraction;
    opt_ext.boundary_policy = "extend";
    auto r_ext = fastlowess::Lowess(opt_ext).fit(linear_x, linear_y).value();

    fastlowess::LowessOptions opt_ref;
    opt_ref.fraction = k_boundary_fraction;
    opt_ref.boundary_policy = "reflect";
    auto r_ref = fastlowess::Lowess(opt_ref).fit(linear_x, linear_y).value();

    fastlowess::LowessOptions opt_zero;
    opt_zero.fraction = k_boundary_fraction;
    opt_zero.boundary_policy = "zero";
    auto r_zr = fastlowess::Lowess(opt_zero).fit(linear_x, linear_y).value();

    std::cout << "Boundary policy comparison:\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " - Extend (Default): first=" << r_ext.yValue(0)
              << ", last=" << r_ext.yValue(k_linear_point_count - 1) << '\n';
    std::cout << " - Reflect:          first=" << r_ref.yValue(0)
              << ", last=" << r_ref.yValue(k_linear_point_count - 1) << '\n';
    std::cout << " - Zero:             first=" << r_zr.yValue(0)
              << ", last=" << r_zr.yValue(k_linear_point_count - 1) << '\n';

    std::cout << "\n=== Batch Smoothing Example Complete ===\n";

  } catch (const std::exception &exception) {
    std::fputs("Error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
