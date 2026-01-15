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

#include "fastlowess.hpp"

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

    fastlowess::StreamingLowess model(opts);
    
    std::cout << "\nProcessing data in chunks..." << std::endl;
    
    size_t chunk_size = opts.chunk_size;
    size_t total_processed = 0;
    
    for (size_t i = 0; i < n; i += chunk_size) {
        size_t current_chunk_len = std::min(chunk_size, n - i);
        std::vector<double> x_chunk(x.begin() + i, x.begin() + i + current_chunk_len);
        std::vector<double> y_chunk(y.begin() + i, y.begin() + i + current_chunk_len);
        
        auto res = model.process_chunk(x_chunk, y_chunk);
        total_processed += res.size();
        
        if (i % 2000 == 0) {
            std::cout << "  Processed " << i << " points..." << std::endl;
        }
    }
    
    auto final_res = model.finalize();
    total_processed += final_res.size();

    std::cout << "\nStreaming completed:" << std::endl;
    std::cout << "  Total points smoothed: " << total_processed << std::endl;

    // Show sample of final results
    if (final_res.size() > 0) {
        std::cout << "\nSample from final chunk:" << std::endl;
        std::cout << "  x=" << final_res.x(0) << " y=" << final_res.y(0) << std::endl;
    }

  } catch (const fastlowess::LowessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n=== Example completed successfully ===" << std::endl;
  return 0;
}
