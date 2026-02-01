#include "../../bindings/cpp/include/fastlowess.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace fastlowess;

// Helper to check approximate equality
bool is_approx(double a, double b, double epsilon = 1e-10) {
  if (std::isnan(a) && std::isnan(b))
    return true;
  return std::abs(a - b) < epsilon;
}

void assert_approx(double a, double b, const std::string &msg = "") {
  if (!is_approx(a, b)) {
    std::cerr << "Assertion failed: " << a << " != " << b << " " << msg
              << std::endl;
    std::exit(1);
  }
}

void assert_approx(double a, double b, double epsilon) {
  if (!is_approx(a, b, epsilon)) {
    std::cerr << "Assertion failed: " << a << " != " << b << " (eps=" << epsilon
              << ")" << std::endl;
    std::exit(1);
  }
}

void assert_true(bool cond, const std::string &msg = "") {
  if (!cond) {
    std::cerr << "Assertion failed " << msg << std::endl;
    std::exit(1);
  }
}

// TestLowess
void test_basic_smooth() {
  std::cout << "Running test_basic_smooth..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

  LowessOptions opts;
  opts.fraction = 0.5;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  assert_true(result.valid(), "Result should be valid");
  assert_true(result.y_vector().size() == 5, "Output length mismatch");
  assert_true(result.x_vector().size() == 5, "X length mismatch");
  assert_approx(result.fraction_used(), 0.5);
}

void test_basic_smooth_serial() {
  std::cout << "Running test_basic_smooth_serial..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.parallel = false;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  assert_true(result.valid());
  assert_true(result.y_vector().size() == 5);
}

void test_lowess_with_diagnostics() {
  std::cout << "Running test_lowess_with_diagnostics..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.return_diagnostics = true;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  auto diag = result.diagnostics();
  assert_true(diag.rmse >= 0, "RMSE negative");
  assert_true(diag.mae >= 0, "MAE negative");
  assert_true(diag.r_squared >= 0 && diag.r_squared <= 1, "R2 out of range");
}

void test_lowess_with_residuals() {
  std::cout << "Running test_lowess_with_residuals..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {2.0, 4.1, 5.9, 8.2, 9.8};

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.return_residuals = true;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  assert_true(result.residuals().size() == 5, "Residuals missing");
}

void test_lowess_with_robustness_weights() {
  std::cout << "Running test_lowess_with_robustness_weights..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {2.0, 4.1, 100.0, 8.2, 9.8}; // Outlier

  LowessOptions opts;
  opts.fraction = 0.7;
  opts.iterations = 3;
  opts.return_robustness_weights = true;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  auto rw = result.robustness_weights();
  assert_true(rw.size() == 5);
  for (double w : rw) {
    assert_true(w >= 0 && w <= 1, "Weight out of range");
  }
}

void test_lowess_with_confidence_intervals() {
  std::cout << "Running test_lowess_with_confidence_intervals..." << std::endl;
  std::vector<double> x(20);
  std::vector<double> y(20);
  for (int i = 0; i < 20; ++i) {
    x[i] = i * (10.0 / 19.0);
    y[i] = 2 * x[i]; // simple linear
  }

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.confidence_intervals = 0.95;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  auto cl = result.confidence_lower();
  auto cu = result.confidence_upper();
  assert_true(cl.size() == 20);
  assert_true(cu.size() == 20);
  for (size_t i = 0; i < 20; ++i) {
    assert_true(cl[i] <= cu[i], "Lower > Upper confidence");
  }
}

void test_lowess_with_prediction_intervals() {
  std::cout << "Running test_lowess_with_prediction_intervals..." << std::endl;
  std::vector<double> x(20);
  std::vector<double> y(20);
  for (int i = 0; i < 20; ++i) {
    x[i] = i * (10.0 / 19.0);
    y[i] = 2 * x[i];
  }

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.prediction_intervals = 0.95;
  Lowess lowess(opts);
  auto result = lowess.fit(x, y).value();

  assert_true(result.prediction_lower().size() == 20);
  assert_true(result.prediction_upper().size() == 20);
}

void test_lowess_reuse() {
  std::cout << "Running test_lowess_reuse..." << std::endl;
  std::vector<double> x1 = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y1 = {2.0, 4.1, 5.9, 8.2, 9.8};
  std::vector<double> x2 = {10.0, 20.0, 30.0, 40.0, 50.0};
  std::vector<double> y2 = {20.0, 40.0, 60.0, 80.0, 100.0};

  LowessOptions opts;
  opts.fraction = 0.5;
  opts.return_diagnostics = true;
  Lowess lowess(opts);

  auto r1 = lowess.fit(x1, y1).value();
  auto r2 = lowess.fit(x2, y2).value();

  assert_true(r1.y_vector().size() == 5);
  assert_true(r2.y_vector().size() == 5);
}

// TestStreamingLowess
void test_streaming_returns_all_points() {
  std::cout << "Running test_streaming_returns_all_points..." << std::endl;
  int n = 100;
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i * (100.0 / 99.0);
    y[i] = 2 * x[i] + 1;
  }

  StreamingOptions opts;
  opts.fraction = 0.3;
  opts.chunk_size = 5000; // > n
  StreamingLowess stream(opts);

  // Cannot move out here directly if we want to combine, actually we can since
  // y_vector() copies
  auto val1 = stream.process_chunk(x, y).value();
  auto val2 = stream.finalize().value();

  assert_true(val1.y_vector().size() + val2.y_vector().size() == n,
              "Total points mismatch");
}

void test_streaming_basic() {
  std::cout << "Running test_streaming_basic..." << std::endl;
  int n = 2000;
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i * (1000.0 / 1999.0);
    y[i] = std::sin(x[i] / 100.0);
  }

  StreamingOptions opts;
  opts.fraction = 0.1;
  opts.chunk_size = 1000;
  StreamingLowess stream(opts);

  auto r1 = stream.process_chunk(x, y).value();
  auto r2 = stream.finalize().value();
}

void test_streaming_accuracy() {
  std::cout << "Running test_streaming_accuracy..." << std::endl;
  int n = 200;
  std::vector<double> x(n), y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = i * (100.0 / 199.0);
    y[i] = 2 * x[i] + 1;
  }

  // Streaming
  StreamingOptions sopts;
  sopts.fraction = 0.5;
  sopts.chunk_size = 1000;
  StreamingLowess stream(sopts);
  auto val1 = stream.process_chunk(x, y).value();
  auto val2 = stream.finalize().value();

  std::vector<double> stream_y;
  auto y1 = val1.y_vector();
  stream_y.insert(stream_y.end(), y1.begin(), y1.end());
  auto y2 = val2.y_vector();
  stream_y.insert(stream_y.end(), y2.begin(), y2.end());

  // Batch
  LowessOptions bopts;
  bopts.fraction = 0.5;
  Lowess batch(bopts);
  auto bres = batch.fit(x, y).value();
  auto by = bres.y_vector();

  assert_true(stream_y.size() == by.size());
  for (size_t i = 0; i < n; ++i) {
    assert_approx(stream_y[i], by[i], 1e-10);
  }
}

// TestOnlineLowess
void test_online_basic() {
  std::cout << "Running test_online_basic..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
  std::vector<double> y = {2.0,  4.0,  6.0,  8.0,  10.0,
                           12.0, 14.0, 16.0, 18.0, 20.0};

  OnlineOptions opts;
  opts.fraction = 0.5;
  opts.window_capacity = 10;
  opts.min_points = 3;
  OnlineLowess online(opts);

  int points_out = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    std::vector<double> xi = {x[i]};
    std::vector<double> yi = {y[i]};
    // Result is rvalue from temporary Expected, needs move or const ref
    const auto &res = online.add_points(xi, yi).value();
    if (!res.y_vector().empty()) {
      points_out++;
    }
  }
  assert_true(points_out > 0);
}

// TestErrorHandling
void test_mismatched_lengths() {
  std::cout << "Running test_mismatched_lengths..." << std::endl;
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {2.0, 4.0};

  LowessOptions opts;
  Lowess lowess(opts);
  try {
    lowess.fit(x, y).value();
    assert_true(false, "Should have thrown");
  } catch (const std::exception &e) {
    // Expected
  }

  // Also test checking has_value()
  auto res = lowess.fit(x, y);
  assert_true(!res.has_value());
  assert_true(!res.error().empty());
}

int main() {
  try {
    test_basic_smooth();
    test_basic_smooth_serial();
    test_lowess_with_diagnostics();
    test_lowess_with_residuals();
    test_lowess_with_robustness_weights();
    test_lowess_with_confidence_intervals();
    test_lowess_with_prediction_intervals();
    test_lowess_reuse();

    test_streaming_returns_all_points();
    test_streaming_basic();
    test_streaming_accuracy();

    test_online_basic();

    test_mismatched_lengths();

    std::cout << "All C++ tests passed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
