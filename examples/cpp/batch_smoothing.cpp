/**
 * @file batch_smoothing.cpp
 * @brief Batch LOWESS smoothing example
 *
 * Demonstrates basic batch smoothing with confidence intervals.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../include/fastlowess.hpp"

int main() {
  std::cout << "=== Batch LOWESS Smoothing Example ===" << std::endl;

  // Generate synthetic data: y = sin(x) + noise
  const size_t n = 100;
  std::vector<double> x(n), y(n);

  std::mt19937 gen(42);
  std::normal_distribution<> noise(0, 0.2);

  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<double>(i) / 10.0;
    y[i] = std::sin(x[i]) + noise(gen);
  }

  std::cout << "Generated " << n << " noisy data points" << std::endl;

  // Basic smoothing
  try {
    fastlowess::LowessOptions opts;
    opts.fraction = 0.3;
    opts.iterations = 3;
    opts.return_residuals = true;
    opts.return_diagnostics = true;
    opts.confidence_intervals = 0.95;

    auto result = fastlowess::smooth(x, y, opts);

    std::cout << "\nSmoothing completed:" << std::endl;
    std::cout << "  Points: " << result.size() << std::endl;
    std::cout << "  Fraction used: " << result.fraction_used() << std::endl;
    std::cout << "  Iterations: " << result.iterations_used() << std::endl;

    // Show diagnostics
    auto diag = result.diagnostics();
    if (diag.has_value()) {
      std::cout << "\nDiagnostics:" << std::endl;
      std::cout << "  RMSE: " << diag.rmse << std::endl;
      std::cout << "  R²: " << diag.r_squared << std::endl;
      std::cout << "  AIC: " << diag.aic << std::endl;
    }

    // Show first 5 points
    std::cout << "\nFirst 5 smoothed points:" << std::endl;
    auto ci_lower = result.confidence_lower();
    auto ci_upper = result.confidence_upper();

    for (size_t i = 0; i < 5 && i < result.size(); ++i) {
      std::cout << "  x=" << result.x(i) << " y=" << result.y(i);
      if (!ci_lower.empty()) {
        std::cout << " CI=[" << ci_lower[i] << ", " << ci_upper[i] << "]";
      }
      std::cout << std::endl;
    }

  } catch (const fastlowess::LowessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== Example completed successfully ===" << std::endl;
  return 0;
}
