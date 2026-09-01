\page cross_validation Cross-Validation

# Cross-Validation

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `fraction`) by evaluating performance on held-out data.

![Cross-Validation](cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

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


    fastlowess::LowessOptions opts;
    opts.cv_fractions = {0.2, 0.3, 0.5, 0.7};
    opts.cv_method = "kfold";
    opts.cv_k = 5;

    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    std::cout << "Selected fraction: " << result.fraction_used() << std::endl;

    return 0;
}
```

```output
Selected fraction: 0.2
```

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

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

    fastlowess::LowessOptions cv_opts;
    cv_opts.cv_method = "loocv";
    cv_opts.cv_fractions = {0.2, 0.3, 0.5, 0.7};
    fastlowess::Lowess model(cv_opts);
    auto result = model.fit(x, y).value();

    std::cout << "Fraction used: " << result.fraction_used() << "\n";
    return 0;
}
```

```output
Fraction used: 0.2
```

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

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


    fastlowess::LowessOptions opts;
    opts.cv_fractions = {0.3, 0.5, 0.7};
    opts.cv_method = "kfold";
    opts.cv_k = 5;
    opts.cv_seed = 42;

    fastlowess::Lowess model(opts);
    auto result = model.fit(x, y).value();

    std::cout << "Fraction used: " << result.fraction_used() << "\n";
    return 0;
}
```

```output
Fraction used: 0.7
```

---

## Comparison

| Method | Folds | Speed | Variance | Bias |
| --- | --- | --- | --- | --- |
| **KFold(5)** | 5 | Fast | Moderate | Low |
| **KFold(10)** | 10 | Medium | Lower | Lower |
| **LOOCV** | N | Slow | Lowest | Lowest |

> **Recommendation:** Use **5-fold** or **10-fold** CV for most applications. LOOCV is only worth it for small datasets (N < 100).

<hr>

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

```text
MSE = mean((y_true - y_pred)^2)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

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

    // Example output
    fastlowess::LowessOptions cv_opts;
    cv_opts.cv_fractions = {0.1, 0.3, 0.5, 0.7};
    cv_opts.cv_method = "kfold";
    cv_opts.cv_k = 5;
    fastlowess::Lowess model(cv_opts);
    auto result = model.fit(x, y).value();

    // Fraction  | CV Score (MSE)
    // 0.1       | 0.0542  ← Undersmoothed
    // 0.3       | 0.0231  ← Best
    // 0.5       | 0.0298
    // 0.7       | 0.0412  ← Oversmoothed

    std::cout << "Fraction used: " << result.fraction_used() << "\n";
    return 0;
}
```

```output
Fraction used: 0.3
```

The fraction with **lowest CV score** is automatically selected.

---

## Availability

> **Batch Mode Only:** Cross-validation is only available in **Batch** mode.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| K-Fold CV | ✓ | ✗ | ✗ |
| LOOCV | ✓ | ✗ | ✗ |

---

## Best Practices

1. **Test a range**: Include fractions from 0.1 to 0.9
2. **Use enough folds**: 5-10 folds balance speed and accuracy
3. **Set a seed**: For reproducible results
4. **Check the curve**: CV optimizes MSE, but visual inspection matters
