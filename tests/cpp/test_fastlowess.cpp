#include "../../bindings/cpp/include/fastlowess.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double k_default_epsilon = 1e-10;
constexpr double k_basic_fraction = 0.5;
constexpr double k_robust_fraction = 0.7;
constexpr double k_confidence_level = 0.95;
constexpr double k_streaming_return_all_fraction = 0.3;
constexpr double k_streaming_basic_fraction = 0.1;
constexpr double k_linear_slope = 2.0;
constexpr double k_linear_intercept = 1.0;
constexpr double k_zero_intercept = 0.0;
constexpr double k_interval_max_x = 10.0;
constexpr double k_streaming_return_all_max_x = 100.0;
constexpr double k_streaming_basic_max_x = 1000.0;
constexpr double k_streaming_basic_sine_divisor = 100.0;
constexpr double k_streaming_accuracy_max_x = 100.0;
constexpr std::size_t k_small_sample_size = 5U;
constexpr std::size_t k_interval_point_count = 20U;
constexpr std::size_t k_streaming_return_all_point_count = 100U;
constexpr std::size_t k_streaming_return_all_chunk_size = 5000U;
constexpr std::size_t k_streaming_basic_point_count = 2000U;
constexpr std::size_t k_streaming_basic_chunk_size = 1000U;
constexpr std::size_t k_streaming_accuracy_point_count = 200U;
constexpr std::size_t k_streaming_accuracy_chunk_size = 1000U;
constexpr std::size_t k_online_window_capacity = 10U;
constexpr std::size_t k_online_min_points = 3U;
constexpr int k_robust_iterations = 3;
constexpr std::size_t k_cw_uniform_point_count = 20U;
constexpr std::size_t k_cw_outlier_point_count = 10U;
constexpr std::size_t k_cw_spike_point_count = 15U;
constexpr std::size_t k_cw_outlier_idx = 5U;
constexpr std::size_t k_cw_spike_idx = 7U;
constexpr double k_cw_uniform_fraction = 0.4;
constexpr double k_cw_outlier_fraction = 0.5;
constexpr double k_cw_spike_fraction = 0.6;
constexpr double k_cw_high_weight = 100.0;
constexpr double k_cw_spike_value = 10.0;
constexpr double k_cw_outlier_value = 100.0;
constexpr double k_cw_slope = 2.0;

constexpr std::array<double, k_small_sample_size> k_sample_x_values = {
    1.0, 2.0, 3.0, 4.0, 5.0};
constexpr std::array<double, k_small_sample_size> k_sample_y_values = {
    2.0, 4.1, 5.9, 8.2, 9.8};
constexpr std::array<double, k_small_sample_size> k_outlier_y_values = {
    2.0, 4.1, 100.0, 8.2, 9.8};
constexpr std::array<double, k_small_sample_size> k_reuse_x_values = {
    10.0, 20.0, 30.0, 40.0, 50.0};
constexpr std::array<double, k_small_sample_size> k_reuse_y_values = {
    20.0, 40.0, 60.0, 80.0, 100.0};
constexpr std::array<double, k_online_window_capacity> k_online_x_values = {
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
constexpr std::array<double, k_online_window_capacity> k_online_y_values = {
    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0};

struct SeriesData {
  std::vector<double> x_values;
  std::vector<double> y_values;
};

struct LinearSeriesSpec {
  std::size_t point_count;
  double max_x_value;
  double slope;
  double intercept;
};

struct SineSeriesSpec {
  std::size_t point_count;
  double max_x_value;
  double divisor;
};

template <std::size_t ElementCount>
std::vector<double> toVector(const std::array<double, ElementCount> &values) {
  return {values.begin(), values.end()};
}

bool isApprox(double actual_value, double expected_value,
              double epsilon = k_default_epsilon) {
  if (std::isnan(actual_value) && std::isnan(expected_value)) {
    return true;
  }
  return std::abs(actual_value - expected_value) < epsilon;
}

void assertApprox(double actual_value, double expected_value,
                  const std::string &message = "") {
  if (!isApprox(actual_value, expected_value)) {
    std::cerr << "Assertion failed: " << actual_value
              << " != " << expected_value << ' ' << message << '\n';
    std::exit(1);
  }
}

void assertApprox(double actual_value, double expected_value, double epsilon) {
  if (!isApprox(actual_value, expected_value, epsilon)) {
    std::cerr << "Assertion failed: " << actual_value
              << " != " << expected_value << " (eps=" << epsilon << ")\n";
    std::exit(1);
  }
}

void assertTrue(bool condition, const std::string &message = "") {
  if (!condition) {
    std::cerr << "Assertion failed " << message << '\n';
    std::exit(1);
  }
}

SeriesData makeLinearData(const LinearSeriesSpec &specification) {
  SeriesData data;
  data.x_values.resize(specification.point_count);
  data.y_values.resize(specification.point_count);

  const double step = specification.max_x_value /
                      static_cast<double>(specification.point_count - 1U);
  for (std::size_t point_index = 0; point_index < specification.point_count;
       ++point_index) {
    const double x_value = static_cast<double>(point_index) * step;
    data.x_values[point_index] = x_value;
    data.y_values[point_index] =
        (specification.slope * x_value) + specification.intercept;
  }

  return data;
}

SeriesData makeSineData(const SineSeriesSpec &specification) {
  SeriesData data;
  data.x_values.resize(specification.point_count);
  data.y_values.resize(specification.point_count);

  const double step = specification.max_x_value /
                      static_cast<double>(specification.point_count - 1U);
  for (std::size_t point_index = 0; point_index < specification.point_count;
       ++point_index) {
    const double x_value = static_cast<double>(point_index) * step;
    data.x_values[point_index] = x_value;
    data.y_values[point_index] = std::sin(x_value / specification.divisor);
  }

  return data;
}

void testBasicSmooth() {
  std::cout << "Running testBasicSmooth...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto sample_y_values = toVector(k_sample_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(sample_x_values, sample_y_values).value();

  assertTrue(result.valid(), "Result should be valid");
  assertTrue(result.y_vector().size() == k_small_sample_size,
             "Output length mismatch");
  assertTrue(result.x_vector().size() == k_small_sample_size,
             "X length mismatch");
  assertApprox(result.fraction_used(), k_basic_fraction);
}

void testBasicSmoothSerial() {
  std::cout << "Running testBasicSmoothSerial...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto sample_y_values = toVector(k_sample_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.parallel = false;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(sample_x_values, sample_y_values).value();

  assertTrue(result.valid(), "Serial result should be valid");
  assertTrue(result.y_vector().size() == k_small_sample_size,
             "Serial output length mismatch");
}

void testLowessWithDiagnostics() {
  std::cout << "Running testLowessWithDiagnostics...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto sample_y_values = toVector(k_sample_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.return_diagnostics = true;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(sample_x_values, sample_y_values).value();

  auto diagnostics = result.diagnostics();
  assertTrue(diagnostics.rmse() >= 0.0, "RMSE negative");
  assertTrue(diagnostics.mae() >= 0.0, "MAE negative");
  assertTrue(diagnostics.r_squared() >= 0.0 && diagnostics.r_squared() <= 1.0,
             "R2 out of range");
}

void testLowessWithResiduals() {
  std::cout << "Running testLowessWithResiduals...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto sample_y_values = toVector(k_sample_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.return_residuals = true;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(sample_x_values, sample_y_values).value();

  assertTrue(result.residuals().size() == k_small_sample_size,
             "Residuals missing");
}

void testLowessWithRobustnessWeights() {
  std::cout << "Running testLowessWithRobustnessWeights...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto outlier_y_values = toVector(k_outlier_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_robust_fraction;
  options.iterations = k_robust_iterations;
  options.return_robustness_weights = true;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(sample_x_values, outlier_y_values).value();

  auto robustness_weights = result.robustness_weights();
  assertTrue(robustness_weights.size() == k_small_sample_size,
             "Robustness weight count mismatch");
  for (const double weight_value : robustness_weights) {
    assertTrue(weight_value >= 0.0 && weight_value <= 1.0,
               "Weight out of range");
  }
}

void testLowessWithConfidenceIntervals() {
  std::cout << "Running testLowessWithConfidenceIntervals...\n";

  const SeriesData linear_data =
      makeLinearData({k_interval_point_count, k_interval_max_x, k_linear_slope,
                      k_zero_intercept});

  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.confidence_intervals = k_confidence_level;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(linear_data.x_values, linear_data.y_values).value();

  auto confidence_lower = result.confidence_lower();
  auto confidence_upper = result.confidence_upper();
  assertTrue(confidence_lower.size() == k_interval_point_count,
             "Confidence lower size mismatch");
  assertTrue(confidence_upper.size() == k_interval_point_count,
             "Confidence upper size mismatch");
  for (std::size_t point_index = 0; point_index < k_interval_point_count;
       ++point_index) {
    assertTrue(confidence_lower[point_index] <= confidence_upper[point_index],
               "Lower > Upper confidence");
  }
}

void testLowessWithPredictionIntervals() {
  std::cout << "Running testLowessWithPredictionIntervals...\n";

  const SeriesData linear_data =
      makeLinearData({k_interval_point_count, k_interval_max_x, k_linear_slope,
                      k_zero_intercept});

  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.prediction_intervals = k_confidence_level;
  fastlowess::Lowess lowess(options);
  auto result = lowess.fit(linear_data.x_values, linear_data.y_values).value();

  assertTrue(result.prediction_lower().size() == k_interval_point_count,
             "Prediction lower size mismatch");
  assertTrue(result.prediction_upper().size() == k_interval_point_count,
             "Prediction upper size mismatch");
}

void testLowessReuse() {
  std::cout << "Running testLowessReuse...\n";

  const auto sample_x_values = toVector(k_sample_x_values);
  const auto sample_y_values = toVector(k_sample_y_values);
  const auto reuse_x_values = toVector(k_reuse_x_values);
  const auto reuse_y_values = toVector(k_reuse_y_values);
  fastlowess::LowessOptions options;
  options.fraction = k_basic_fraction;
  options.return_diagnostics = true;
  fastlowess::Lowess lowess(options);

  auto first_result = lowess.fit(sample_x_values, sample_y_values).value();
  auto second_result = lowess.fit(reuse_x_values, reuse_y_values).value();

  assertTrue(first_result.y_vector().size() == k_small_sample_size,
             "First reuse result size mismatch");
  assertTrue(second_result.y_vector().size() == k_small_sample_size,
             "Second reuse result size mismatch");
}

void testStreamingReturnsAllPoints() {
  std::cout << "Running testStreamingReturnsAllPoints...\n";

  const SeriesData linear_data = makeLinearData(
      {k_streaming_return_all_point_count, k_streaming_return_all_max_x,
       k_linear_slope, k_linear_intercept});

  fastlowess::StreamingOptions options;
  options.fraction = k_streaming_return_all_fraction;
  options.chunk_size = k_streaming_return_all_chunk_size;
  fastlowess::StreamingLowess streaming_lowess(options);

  auto first_chunk_result =
      streaming_lowess.process_chunk(linear_data.x_values, linear_data.y_values)
          .value();
  auto final_result = streaming_lowess.finalize().value();

  const std::size_t total_point_count =
      first_chunk_result.y_vector().size() + final_result.y_vector().size();
  assertTrue(total_point_count == k_streaming_return_all_point_count,
             "Total points mismatch");
}

void testStreamingBasic() {
  std::cout << "Running testStreamingBasic...\n";

  const SeriesData sine_data =
      makeSineData({k_streaming_basic_point_count, k_streaming_basic_max_x,
                    k_streaming_basic_sine_divisor});

  fastlowess::StreamingOptions options;
  options.fraction = k_streaming_basic_fraction;
  options.chunk_size = k_streaming_basic_chunk_size;
  fastlowess::StreamingLowess streaming_lowess(options);

  auto first_chunk_result =
      streaming_lowess.process_chunk(sine_data.x_values, sine_data.y_values)
          .value();
  auto final_result = streaming_lowess.finalize().value();

  assertTrue(!first_chunk_result.y_vector().empty() ||
                 !final_result.y_vector().empty(),
             "Streaming basic produced no points");
}

void testStreamingAccuracy() {
  std::cout << "Running testStreamingAccuracy...\n";

  const SeriesData linear_data = makeLinearData(
      {k_streaming_accuracy_point_count, k_streaming_accuracy_max_x,
       k_linear_slope, k_linear_intercept});

  fastlowess::StreamingOptions streaming_options;
  streaming_options.fraction = k_basic_fraction;
  streaming_options.chunk_size = k_streaming_accuracy_chunk_size;
  fastlowess::StreamingLowess streaming_lowess(streaming_options);
  auto first_chunk_result =
      streaming_lowess.process_chunk(linear_data.x_values, linear_data.y_values)
          .value();
  auto final_result = streaming_lowess.finalize().value();

  std::vector<double> streaming_y_values;
  auto first_y_values = first_chunk_result.y_vector();
  streaming_y_values.insert(streaming_y_values.end(), first_y_values.begin(),
                            first_y_values.end());
  auto final_y_values = final_result.y_vector();
  streaming_y_values.insert(streaming_y_values.end(), final_y_values.begin(),
                            final_y_values.end());

  fastlowess::LowessOptions batch_options;
  batch_options.fraction = k_basic_fraction;
  fastlowess::Lowess batch_lowess(batch_options);
  auto batch_result =
      batch_lowess.fit(linear_data.x_values, linear_data.y_values).value();
  auto batch_y_values = batch_result.y_vector();

  assertTrue(streaming_y_values.size() == batch_y_values.size(),
             "Streaming and batch sizes differ");
  for (std::size_t point_index = 0;
       point_index < k_streaming_accuracy_point_count; ++point_index) {
    assertApprox(streaming_y_values[point_index], batch_y_values[point_index],
                 k_default_epsilon);
  }
}

void testOnlineBasic() {
  std::cout << "Running testOnlineBasic...\n";

  fastlowess::OnlineOptions options;
  options.fraction = k_basic_fraction;
  options.window_capacity = k_online_window_capacity;
  options.min_points = static_cast<int>(k_online_min_points);
  fastlowess::OnlineLowess online_lowess(options);

  int point_count_with_output = 0;
  for (std::size_t point_index = 0; point_index < k_online_x_values.size();
       ++point_index) {
    auto result = online_lowess
                      .add_point(k_online_x_values[point_index],
                                 k_online_y_values[point_index])
                      .value();
    if (result.has_value()) {
      ++point_count_with_output;
    }
  }
  assertTrue(point_count_with_output > 0,
             "Online smoothing produced no points");
}

void testMismatchedLengths() {
  std::cout << "Running testMismatchedLengths...\n";

  const std::vector<double> mismatched_x_values = {1.0, 2.0, 3.0};
  const std::vector<double> mismatched_y_values = {2.0, 4.0};

  const fastlowess::LowessOptions options;
  fastlowess::Lowess lowess(options);
  bool threw_exception = false;
  std::string exception_message;
  try {
    lowess.fit(mismatched_x_values, mismatched_y_values).value();
  } catch (const std::exception &exception) {
    threw_exception = true;
    exception_message = exception.what();
  }

  assertTrue(threw_exception, "Should have thrown");
  assertTrue(!exception_message.empty(), "Exception message should be present");

  auto result = lowess.fit(mismatched_x_values, mismatched_y_values);
  assertTrue(!result.has_value(),
             "Expected error result for mismatched lengths");
  assertTrue(!result.error().empty(), "Expected non-empty error message");
}

void testCustomWeightsUniformMatchesNoWeights() {
  std::cout << "Running testCustomWeightsUniformMatchesNoWeights...\n";

  const std::size_t point_count = k_cw_uniform_point_count;
  std::vector<double> x_values(point_count);
  std::vector<double> y_values(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    x_values[i] = static_cast<double>(i) * k_cw_outlier_fraction;
    y_values[i] = std::sin(x_values[i]);
  }
  const std::vector<double> weights(point_count, 1.0);

  fastlowess::LowessOptions opts;
  opts.fraction = k_cw_uniform_fraction;
  opts.iterations = 2;

  fastlowess::Lowess model(opts);
  auto result_no_w = model.fit(x_values, y_values).value();
  auto result_unit_w = model.fit(x_values, y_values, weights).value();

  const auto y_no_w = result_no_w.y_vector();
  const auto y_unit_w = result_unit_w.y_vector();

  for (std::size_t i = 0; i < point_count; ++i) {
    assertApprox(y_no_w[i], y_unit_w[i], k_default_epsilon);
  }
}

void testCustomWeightsZeroWeightReducesOutlierInfluence() {
  std::cout
      << "Running testCustomWeightsZeroWeightReducesOutlierInfluence...\n";

  const std::size_t point_count = k_cw_outlier_point_count;
  std::vector<double> x_values(point_count);
  std::vector<double> y_values(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    x_values[i] = static_cast<double>(i);
    y_values[i] = x_values[i] * k_cw_slope;
  }
  y_values[k_cw_outlier_idx] = k_cw_outlier_value; // outlier

  fastlowess::LowessOptions opts;
  opts.fraction = k_cw_outlier_fraction;
  opts.iterations = 0;

  fastlowess::Lowess model(opts);
  auto result_no_w = model.fit(x_values, y_values).value();

  std::vector<double> weights(point_count, 1.0);
  weights[k_cw_outlier_idx] = 0.0;
  auto result_zero_w = model.fit(x_values, y_values, weights).value();

  const double true_val = static_cast<double>(k_cw_outlier_idx) * k_cw_slope;
  const double err_no_w =
      std::abs(result_no_w.y_value(k_cw_outlier_idx) - true_val);
  const double err_zero_w =
      std::abs(result_zero_w.y_value(k_cw_outlier_idx) - true_val);

  assertTrue(err_zero_w < err_no_w,
             "zero weight at outlier should reduce fitting error");
}

void testCustomWeightsHighWeightPullsFit() {
  std::cout << "Running testCustomWeightsHighWeightPullsFit...\n";

  const std::size_t point_count = k_cw_spike_point_count;
  std::vector<double> x_values(point_count);
  std::vector<double> y_values(point_count, 0.0);
  for (std::size_t i = 0; i < point_count; ++i) {
    x_values[i] = static_cast<double>(i);
  }
  y_values[k_cw_spike_idx] = k_cw_spike_value; // spike

  fastlowess::LowessOptions opts;
  opts.fraction = k_cw_spike_fraction;
  opts.iterations = 0;

  std::vector<double> weights_high(point_count, 1.0);
  weights_high[k_cw_spike_idx] = k_cw_high_weight;

  fastlowess::Lowess model(opts);
  auto result_high = model.fit(x_values, y_values, weights_high).value();
  auto result_equal = model.fit(x_values, y_values).value();

  assertTrue(result_high.y_value(k_cw_spike_idx) >
                 result_equal.y_value(k_cw_spike_idx),
             "high weight at spike should pull fit up");
}

} // namespace

int main() {
  try {
    testBasicSmooth();
    testBasicSmoothSerial();
    testLowessWithDiagnostics();
    testLowessWithResiduals();
    testLowessWithRobustnessWeights();
    testLowessWithConfidenceIntervals();
    testLowessWithPredictionIntervals();
    testLowessReuse();
    testStreamingReturnsAllPoints();
    testStreamingBasic();
    testStreamingAccuracy();
    testOnlineBasic();
    testMismatchedLengths();
    testCustomWeightsUniformMatchesNoWeights();
    testCustomWeightsZeroWeightReducesOutlierInfluence();
    testCustomWeightsHighWeightPullsFit();

    std::cout << "All C++ tests passed!\n";
  } catch (const std::exception &exception) {
    std::fputs("Test failed with exception: ", stderr);
    std::fputs(exception.what(), stderr);
    std::fputc('\n', stderr);
    return 1;
  }
  return 0;
}
