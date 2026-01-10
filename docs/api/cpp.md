<!-- markdownlint-disable MD024 -->
# C++ API Reference

C and C++ bindings for fastLowess.

## Installation

### From Source

```bash
git clone https://github.com/thisisamirv/lowess-project
cd lowess-project/bindings/cpp

# Build the library
cargo build --release

# Headers are at: include/fastlowess.h (C) and include/fastlowess.hpp (C++)
# Library is at: target/release/libfastlowess_cpp.so (Linux)
#                target/release/libfastlowess_cpp.dylib (macOS)
#                target/release/fastlowess_cpp.dll (Windows)
```

---

## Quick Start

```cpp
#include <vector>
#include <iostream>
#include "fastlowess.hpp"

int main() {
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {2.1, 3.9, 6.2, 8.0, 10.1};

    // Smooth with default options
    auto result = fastlowess::smooth(x, y);

    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result.x(i) << " -> " << result.y(i) << std::endl;
    }
    return 0;
}
```

---

## API

### `fastlowess::smooth()`

Batch LOWESS smoothing.

```cpp
LowessResult smooth(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const LowessOptions& options = {}
);
```

### `fastlowess::streaming()`

Streaming LOWESS for large datasets.

```cpp
LowessResult streaming(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const StreamingOptions& options = {}
);
```

### `fastlowess::online()`

Online LOWESS with sliding window.

```cpp
LowessResult online(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const OnlineOptions& options = {}
);
```

---

## Options

### `LowessOptions`

| Field                  | Type          | Default      | Description                       |
|------------------------|---------------|--------------|-----------------------------------|
| `fraction`             | `double`      | `0.67`       | Smoothing fraction                |
| `iterations`           | `int`         | `3`          | Robustness iterations             |
| `weight_function`      | `std::string` | `"tricube"`  | Weight function                   |
| `robustness_method`    | `std::string` | `"bisquare"` | Robustness method                 |
| `confidence_intervals` | `double`      | `NAN`        | Confidence level (NaN = disabled) |
| `return_diagnostics`   | `bool`        | `false`      | Return fit diagnostics            |
| `parallel`             | `bool`        | `false`      | Enable parallel processing        |

### `StreamingOptions`

Extends `LowessOptions` with:

| Field        | Type  | Default | Description                        |
|--------------|-------|---------|------------------------------------|
| `chunk_size` | `int` | `5000`  | Points per chunk                   |
| `overlap`    | `int` | `-1`    | Overlap between chunks (-1 = auto) |

### `OnlineOptions`

Extends `LowessOptions` with:

| Field             | Type          | Default  | Description                          |
|-------------------|---------------|----------|--------------------------------------|
| `window_capacity` | `int`         | `1000`   | Sliding window size                  |
| `min_points`      | `int`         | `2`      | Minimum points for smoothing         |
| `update_mode`     | `std::string` | `"full"` | Update mode: "full" or "incremental" |

---

## LowessResult

RAII wrapper with automatic memory management.

```cpp
class LowessResult {
public:
    size_t size() const;              // Number of points
    bool valid() const;               // Check if result is valid
    
    double x(size_t i) const;         // Access x value
    double y(size_t i) const;         // Access smoothed y value
    
    std::vector<double> x_vector() const;
    std::vector<double> y_vector() const;
    std::vector<double> residuals() const;
    std::vector<double> confidence_lower() const;
    std::vector<double> confidence_upper() const;
    
    double fraction_used() const;
    int iterations_used() const;
    Diagnostics diagnostics() const;
};
```

---

## Error Handling

Errors throw `fastlowess::LowessError`:

```cpp
try {
    auto result = fastlowess::smooth(x, y);
} catch (const fastlowess::LowessError& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}
```
