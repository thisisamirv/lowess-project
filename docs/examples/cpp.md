# C++ Examples

Complete C++ examples demonstrating the fastLowess C++ bindings with modern C++ features.

## Batch Smoothing

Process complete datasets with the idiomatic C++ wrapper.

```cpp
--8<-- "../../examples/cpp/batch_smoothing.cpp"
```

[:material-download: Download batch_smoothing.cpp](https://github.com/thisisamirv/lowess-project/blob/main/examples/cpp/batch_smoothing.cpp)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks.

```cpp
--8<-- "../../examples/cpp/streaming_smoothing.cpp"
```

[:material-download: Download streaming_smoothing.cpp](https://github.com/thisisamirv/lowess-project/blob/main/examples/cpp/streaming_smoothing.cpp)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data.

```cpp
--8<-- "../../examples/cpp/online_smoothing.cpp"
```

[:material-download: Download online_smoothing.cpp](https://github.com/thisisamirv/lowess-project/blob/main/examples/cpp/online_smoothing.cpp)

---

## Building the Examples

```bash
# Build the C++ bindings
make cpp

# The examples are built as part of the bindings
# Or compile manually:
g++ -std=c++20 -I bindings/cpp/include \
    ../../examples/cpp/batch_smoothing.cpp \
    -L target/release -lfastlowess_cpp \
    -o batch_smoothing
```

## Quick Start

```cpp
#include <fastlowess.hpp>
#include <iostream>
#include <vector>

int main() {
    // Generate sample data
    std::vector<double> x(100), y(100);
    for (size_t i = 0; i < 100; ++i) {
        x[i] = i * 0.1;
        y[i] = std::sin(x[i]) + 0.1;
    }

    // Configure options
    fastlowess::LowessOptions options;
    options.fraction = 0.3;
    options.iterations = 3;
    options.confidence_intervals = 0.95;
    options.return_diagnostics = true;

    // Smooth
    try {
        auto result = fastlowess::smooth(x, y, options);
        
        std::cout << "R²: " << result.diagnostics().r_squared << std::endl;
        
        // Access smoothed values
        auto smoothed = result.y_vector();
    } catch (const fastlowess::LowessError& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

## Features

The C++ bindings provide:

- **RAII memory management** - Resources automatically freed
- **STL container support** - `std::vector<double>` for all arrays
- **Exception-based errors** - `fastlowess::LowessError` for error handling
- **Modern C++ idioms** - Designated initializers, move semantics
