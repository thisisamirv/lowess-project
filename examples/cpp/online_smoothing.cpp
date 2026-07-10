/**
 * @file online_smoothing.cpp
 * @brief Online LOWESS smoothing example
 *
 * Demonstrates sliding window smoothing for real-time data.
 */

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "../../bindings/cpp/include/fastlowess.hpp"

namespace {

constexpr size_t k_point_count = 200;
constexpr unsigned int k_random_seed = 42;
constexpr double k_noise_std_dev = 0.3;
constexpr double k_trend_slope = 0.1;
constexpr double k_seasonal_amplitude = 5.0;
constexpr double k_seasonal_period_divisor = 20.0;
constexpr double k_fraction = 0.5;
constexpr int k_window_capacity = 50;
constexpr int k_min_points = 10;
constexpr size_t k_progress_interval = 40;

} // namespace

int main() {
  try {
    std::cout << "=== Online LOWESS Smoothing Example ===\n";

    // Simulate streaming data arrival
    const size_t point_count = k_point_count;
    std::vector<double> x_values(point_count);
    std::vector<double> y_values(point_count);

    std::seed_seq generator_seed = {k_random_seed, k_random_seed, k_random_seed,
                                    k_random_seed};
    std::mt19937 generator(generator_seed);
    std::normal_distribution<> noise(0.0, k_noise_std_dev);

    for (size_t point_index = 0; point_index < point_count; ++point_index) {
      x_values[point_index] = static_cast<double>(point_index);
      // Trend with seasonal component + noise
      y_values[point_index] =
          (k_trend_slope * x_values[point_index]) +
          (k_seasonal_amplitude *
           std::sin(x_values[point_index] / k_seasonal_period_divisor)) +
          noise(generator);
    }

    std::cout << "Generated " << point_count << " streaming data points\n";

    // Online smoothing with sliding window
    fastlowess::OnlineOptions opts;
    opts.fraction = k_fraction;
    opts.iterations = 2;
    opts.window_capacity = k_window_capacity;
    opts.min_points = k_min_points;
    opts.update_mode = "full";

    std::cout << "\nProcessing with window_capacity=" << opts.window_capacity
              << ", min_points=" << opts.min_points << '\n';

    fastlowess::OnlineLowess model(opts);

    std::cout << "\nProcessing data point-by-point...\n";

    size_t total_emitted = 0;
    for (size_t point_index = 0; point_index < point_count; ++point_index) {
      auto output =
          model.add_point(x_values[point_index], y_values[point_index]).value();

      if (output.has_value()) {
        ++total_emitted;
      }

      if (point_index > 0 && point_index % k_progress_interval == 0 &&
          output.has_value()) {
        std::cout << "  t=" << point_index
                  << " original=" << y_values[point_index]
                  << " smoothed=" << output.smoothed() << '\n';
      }
    }

    std::cout << "\nOnline processing completed:\n";
    std::cout << "  Total points with smoothed output: " << total_emitted
              << '\n';

    std::cout << "\n=== Example completed successfully ===\n";

  } catch (const std::exception &exception) {
    std::fputs("Error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
