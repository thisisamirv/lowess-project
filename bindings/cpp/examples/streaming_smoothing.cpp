/**
 * @file streaming_smoothing.cpp
 * @brief Streaming LOWESS smoothing example
 *
 * Demonstrates chunk-based processing for large datasets.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <random>
#include <vector>

#include "../../bindings/cpp/include/fastlowess.hpp"

namespace {

constexpr size_t k_point_count = 10000;
constexpr unsigned int k_random_seed = 42;
constexpr double k_noise_std_dev = 0.5;
constexpr double k_sine_divisor = 10.0;
constexpr double k_scale_divisor = 50.0;
constexpr double k_fraction = 0.1;
constexpr int k_chunk_size = 1000;
constexpr int k_overlap = 100;
constexpr size_t k_progress_interval = 2000;

} // namespace

int main() {
  try {
    std::cout << "=== Streaming LOWESS Smoothing Example ===\n";

    // Generate large synthetic dataset
    const size_t point_count = k_point_count;
    std::vector<double> x_values(point_count);
    std::vector<double> y_values(point_count);

    std::seed_seq generator_seed = {k_random_seed, k_random_seed, k_random_seed,
                                    k_random_seed};
    std::mt19937 generator(generator_seed);
    std::normal_distribution<> noise(0.0, k_noise_std_dev);

    for (size_t point_index = 0; point_index < point_count; ++point_index) {
      x_values[point_index] = static_cast<double>(point_index) / k_overlap;
      y_values[point_index] =
          ((std::sin(x_values[point_index] / k_sine_divisor) *
            x_values[point_index]) /
           k_scale_divisor) +
          noise(generator);
    }

    std::cout << "Generated " << point_count << " data points\n";

    // Streaming smoothing
    fastlowess::StreamingOptions opts;
    opts.fraction = k_fraction;
    opts.iterations = 2;
    opts.chunk_size = k_chunk_size;
    opts.overlap = k_overlap;
    opts.return_diagnostics = true;

    std::cout << "\nProcessing with chunk_size=" << opts.chunk_size
              << ", overlap=" << opts.overlap << '\n';

    fastlowess::StreamingLowess model(opts);

    std::cout << "\nProcessing data in chunks...\n";

    const size_t chunk_size = static_cast<size_t>(opts.chunk_size);
    size_t total_processed = 0;

    for (size_t chunk_start = 0; chunk_start < point_count;
         chunk_start += chunk_size) {
      const size_t current_chunk_len =
          std::min(chunk_size, point_count - chunk_start);
      std::vector<double> x_chunk(current_chunk_len);
      std::vector<double> y_chunk(current_chunk_len);

      std::copy_n(x_values.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                  static_cast<std::ptrdiff_t>(current_chunk_len),
                  x_chunk.begin());
      std::copy_n(y_values.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                  static_cast<std::ptrdiff_t>(current_chunk_len),
                  y_chunk.begin());

      auto res = model.process_chunk(x_chunk, y_chunk).value();
      total_processed += res.size();

      if (chunk_start % k_progress_interval == 0) {
        std::cout << "  Processed " << chunk_start << " points...\n";
      }
    }

    auto final_res = model.finalize().value();
    total_processed += final_res.size();

    std::cout << "\nStreaming completed:\n";
    std::cout << "  Total points smoothed: " << total_processed << '\n';

    // Show sample of final results
    if (final_res.size() > 0) {
      std::cout << "\nSample from final chunk:\n";
      std::cout << "  x=" << final_res.x_value(0)
                << " y=" << final_res.y_value(0) << '\n';
    }

    std::cout << "\n=== Example completed successfully ===\n";

  } catch (const std::exception &exception) {
    std::fputs("Error: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
