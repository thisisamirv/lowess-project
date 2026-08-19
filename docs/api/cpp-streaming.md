# StreamingLowess — C++ API Reference

See also: [fastLowess C++ API Reference](cpp.md)

## Class

### `fastlowess::StreamingLowess`

The `StreamingLowess` class processes data in chunks, suitable for very large datasets or streaming applications.

**Constructor:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    fastlowess::StreamingOptions opts;
    opts.chunk_size = 5;
    fastlowess::StreamingLowess model(opts);

    return 0;
}
```

* `options`: A `StreamingOptions` struct (inherits from `LowessOptions`) with additional `chunk_size`, `overlap`, and `merge_strategy` parameters.

**Methods:**

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::StreamingOptions opts;
    opts.fraction = 0.5;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastlowess::StreamingLowess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    auto partial = model.process_chunk(x1, y1).value();
    std::cout << partial.fraction_used() << std::endl;  // 0.5

    return 0;
}
```

* Processes a chunk of data. Returns partial results.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    fastlowess::StreamingOptions opts;
    opts.fraction = 0.3;
    opts.chunk_size = 50;
    opts.overlap = 10;
    fastlowess::StreamingLowess model(opts);
    std::vector<double> x1(x.begin(), x.begin() + 50), y1(y.begin(), y.begin() + 50);
    std::vector<double> x2(x.begin() + 50, x.end()), y2(y.begin() + 50, y.end());
    model.process_chunk(x1, y1);
    model.process_chunk(x2, y2);
    auto result = model.finalize().value();
    std::cout << result.fraction_used() << std::endl;  // 0.3

    return 0;
}
```

* Finalizes the smoothing process and returns any remaining buffered results.

## Result Structure

### `fastlowess::LowessResult`

Returned (inside `Expected`) by `process_chunk()` and `finalize()`.

| Method | Return Type | Description |
| --- | --- | --- |
| `x_vector()` | `std::vector<double>` | Sorted x values |
| `y_vector()` | `std::vector<double>` | Smoothed y values |
| `fraction_used()` | `double` | Fraction used |
| `iterations_used()` | `int` | Robustness iterations (-1 = N/A) |
| `residuals()` | `std::vector<double>` | Residuals (if `return_residuals`; empty if not) |
| `robustness_weights()` | `std::vector<double>` | Robustness weights (if `return_robustness_weights`; empty if not) |
| `diagnostics()` | `Diagnostics` | Fit metrics — check `has_value()` (if `return_diagnostics`) |
| `dimensions()` | `int` | Number of predictor dimensions |

See [cpp.md](cpp.md) for the full `LowessResult` field reference.

## Options Structure

### `StreamingOptions` (inherits `LowessOptions`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `chunk_size` | `int` | 5000 | Data chunk size |
| `overlap` | `int` | 500 | Overlap between chunks |
| `merge_strategy` | `std::string` | "weighted_average" | Strategy for blending overlap regions |

## Options

### merge_strategy

*See: [Merge Strategies](../user-guide/merge.md)*

* `"weighted_average"` (default; alias: `"weighted"`)
* `"average"` (alias: `"mean"`)
* `"take_first"` (alias: `"first"`)
* `"take_last"` (alias: `"last"`)
