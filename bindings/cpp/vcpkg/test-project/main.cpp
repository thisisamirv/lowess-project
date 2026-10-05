#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fastlowess.hpp>
#include <vector>

int main() try {
  constexpr double k_linear_tolerance = 1e-8;
  const std::vector<double> x_values = {1, 2, 3, 4, 5, 6};
  const std::vector<double> y_values = {3, 5, 7, 9, 11, 13};
  fastlowess::LowessOptions options;
  options.fraction = 1.0;
  options.iterations = 0;
  options.parallel = false;
  options.boundary_policy = "noboundary";
  fastlowess::Lowess model(options);
  const auto result = model.fit(x_values, y_values).value();
  if (!result.valid() || result.size() != y_values.size()) {
    return 1;
  }
  for (std::size_t index = 0; index < y_values.size(); ++index) {
    const double fitted = result.y_value(index);
    if (!std::isfinite(fitted) ||
        std::abs(fitted - y_values[index]) > k_linear_tolerance) {
      return 2;
    }
  }
  return 0;
} catch (const std::exception &) {
  return EXIT_FAILURE;
}