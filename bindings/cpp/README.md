# fastLowess C++ Bindings

C and C++ bindings for the fastLowess library.

## Building

```bash
# Build the library
cargo build --release

# The header file will be generated at include/fastlowess.h
# The shared library will be at target/release/libfastlowess_cpp.so (Linux)
```

## Usage

### C

```c
#include "fastlowess.h"

int main() {
    double x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[] = {2.1, 3.9, 6.2, 8.0, 10.1};

    CppLowessResult result = cpp_lowess_smooth(
        x, y, 5,
        0.67, 3, NAN,     // fraction, iterations, delta
        "tricube",        // weight function
        "bisquare",       // robustness method
        "mad",            // scaling method
        "extend",         // boundary policy
        NAN, NAN,         // confidence/prediction intervals
        0, 0, 0,          // diagnostics, residuals, robustness weights
        "use_local_mean", // zero weight fallback
        NAN,              // auto converge
        NULL, 0,          // cv fractions
        NULL, 5,          // cv method
        0                 // parallel
    );

    if (result.error == NULL) {
        for (size_t i = 0; i < result.n; i++) {
            printf("x=%.2f y=%.2f\n", result.x[i], result.y[i]);
        }
    }

    cpp_lowess_free_result(&result);
    return 0;
}
```

### C++

See `include/fastlowess.hpp` for the idiomatic C++ wrapper.
