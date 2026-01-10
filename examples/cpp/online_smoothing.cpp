/**
 * @file online_smoothing.cpp
 * @brief Online LOWESS smoothing example
 *
 * Demonstrates sliding window smoothing for real-time data.
 */

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../include/fastlowess.hpp"

int main() {
  std::cout << "=== Online LOWESS Smoothing Example ===" << std::endl;

  // Simulate streaming data arrival
  const size_t n = 200;
  std::vector<double> x(n), y(n);

  std::mt19937 gen(42);
  std::normal_distribution<> noise(0, 0.3);

  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<double>(i);
    // Trend with seasonal component + noise
    y[i] = 0.1 * x[i] + 5.0 * std::sin(x[i] / 20.0) + noise(gen);
  }

  std::cout << "Generated " << n << " streaming data points" << std::endl;

  // Online smoothing with sliding window
  try {
    fastlowess::OnlineOptions opts;
    opts.fraction = 0.5;
    opts.iterations = 2;
    opts.window_capacity = 50;
    opts.min_points = 10;
    opts.update_mode = "full";

    std::cout << "\nProcessing with window_capacity=" << opts.window_capacity
              << ", min_points=" << opts.min_points << std::endl;

    auto result = fastlowess::online(x, y, opts);

    std::cout << "\nOnline processing completed:" << std::endl;
    std::cout << "  Points processed: " << result.size() << std::endl;
    std::cout << "  Fraction used: " << result.fraction_used() << std::endl;

    // Show sample of results
    std::cout << "\nSample points (every 20th):" << std::endl;
    for (size_t i = 0; i < result.size(); i += 20) {
      double original = y[i];
      double smoothed = result.y(i);
      std::cout << "  t=" << result.x(i) << " original=" << original
                << " smoothed=" << smoothed << " diff=" << (original - smoothed)
                << std::endl;
    }

  } catch (const fastlowess::LowessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== Example completed successfully ===" << std::endl;
  return 0;
}
