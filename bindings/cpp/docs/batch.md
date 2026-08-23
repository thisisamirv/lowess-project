# Batch Adapter

Standard mode for complete datasets. **Supports all features.**

## When to Use

- Dataset fits in memory
- Need intervals, cross-validation, or diagnostics
- Processing complete files

![Gap Handling](../assets/diagrams/gap_handling.svg)

## Example

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


    fastlowess::Lowess model({
        .fraction = 0.5,
        .iterations = 3,
        .confidence_intervals = 0.95,
        .prediction_intervals = 0.95,
        .return_diagnostics = true,
        .parallel = true
    });
    auto result = model.fit(x, y).value();

    return 0;
}
```

---
