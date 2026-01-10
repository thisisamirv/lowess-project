/**
 * @file streaming_smoothing.cpp
 * @brief Streaming LOWESS smoothing example
 *
 * Demonstrates chunk-based processing for large datasets.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../include/fastlowess.hpp"

int main() {
  std::cout << "=== Streaming LOWESS Smoothing Example ===" << std::endl;

  // Generate large synthetic dataset
  const size_t n = 10000;
  std::vector<double> x(n), y(n);

  std::mt19937 gen(42);
  std::normal_distribution<> noise(0, 0.5);

  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<double>(i) / 100.0;
    y[i] = std::sin(x[i] / 10.0) * x[i] / 50.0 + noise(gen);
  }

  std::cout << "Generated " << n << " data points" << std::endl;

  // Streaming smoothing
  try {
    fastlowess::StreamingOptions opts;
    opts.fraction = 0.1;
    opts.iterations = 2;
    opts.chunk_size = 1000;
    opts.overlap = 100;
    opts.return_diagnostics = true;

    std::cout << "\nProcessing with chunk_size=" << opts.chunk_size
              << ", overlap=" << opts.overlap << std::endl;

    auto result = fastlowess::streaming(x, y, opts);

    std::cout << "\nStreaming completed:" << std::endl;
    std::cout << "  Points processed: " << result.size() << std::endl;
    std::cout << "  Fraction used: " << result.fraction_used() << std::endl;

    // Show diagnostics
    auto diag = result.diagnostics();
    if (diag.has_value()) {
      std::cout << "\nDiagnostics:" << std::endl;
      std::cout << "  RMSE: " << diag.rmse << std::endl;
      std::cout << "  R²: " << diag.r_squared << std::endl;
    }

    // Show sample of results
    std::cout << "\nSample points (every 1000th):" << std::endl;
    for (size_t i = 0; i < result.size(); i += 1000) {
      std::cout << "  x=" << result.x(i) << " y=" << result.y(i) << std::endl;
    }

  } catch (const fastlowess::LowessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== Example completed successfully ===" << std::endl;
  return 0;
}
