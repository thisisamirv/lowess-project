---
title: "Cross-Validation"
weight: 65
---

Automated parameter selection via cross-validation.

## Overview

Cross-validation helps select optimal parameters (especially `fraction`) by evaluating performance on held-out data.

![Cross-Validation](../assets/diagrams/cv_comparison.svg)

---

## K-Fold Cross-Validation

Split data into K folds, train on K-1, validate on 1.

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        int n = 100;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 2 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]) + 0.1;
        }

        Options options = Options.builder()
                .cvMethod("kfold")
                .cvK(5)
                .cvFractions(new double[] { 0.2, 0.3, 0.5, 0.7 })
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);

            // The best fraction was automatically selected
            System.out.println("Selected fraction: " + result.fractionUsed());
            System.out.println("CV scores: " + java.util.Arrays.toString(result.cvScores().orElseThrow()));
        }
    }
}
```

```output
Selected fraction: 0.2
CV scores: [0.2660065925434151, 0.26663733476135043, 0.36243048571062464, 0.4466813477111353]
```

---

## Leave-One-Out (LOOCV)

Each point is held out once. Most thorough but slowest.

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        int n = 100;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 2 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]) + 0.1;
        }

        Options options = Options.builder()
                .cvMethod("loocv")
                .cvFractions(new double[] { 0.2, 0.3, 0.5, 0.7 })
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("Selected fraction (CV): " + result.fractionUsed());
        }
    }
}
```

```output
Selected fraction (CV): 0.2
```

---

## Seeded Randomization

Set a seed for reproducible fold assignments:

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        int n = 100;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 2 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]) + 0.1;
        }

        Options options = Options.builder()
                .cvMethod("kfold")
                .cvK(5)
                .cvFractions(new double[] { 0.3, 0.5, 0.7 })
                .cvSeed(42L)
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("Selected fraction (CV): " + result.fractionUsed());
        }
    }
}
```

```output
Selected fraction (CV): 0.7
```

---

## Comparison

| Method | Folds | Speed | Variance | Bias |
| --- | --- | --- | --- | --- |
| **KFold(5)** | 5 | Fast | Moderate | Low |
| **KFold(10)** | 10 | Medium | Lower | Lower |
| **LOOCV** | N | Slow | Lowest | Lowest |

> **Recommendation:** Use **5-fold** or **10-fold** CV for most applications. LOOCV is only worth it for small datasets (N < 100).

---

## CV Metrics

Cross-validation uses MSE (Mean Squared Error) by default:

```text
MSE = mean((y_true - y_pred)^2)
```

Lower MSE indicates better fit on held-out data.

---

## Interpreting Results

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        int n = 100;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 2 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]) + 0.1;
        }

        Options options = Options.builder()
                .cvMethod("kfold")
                .cvK(5)
                .cvFractions(new double[] { 0.1, 0.3, 0.5, 0.7 })
                .build();

        // Fraction  | CV Score (MSE)
        // 0.1       | 0.0542  <- Undersmoothed
        // 0.3       | 0.0231  <- Best
        // 0.5       | 0.0298
        // 0.7       | 0.0412  <- Oversmoothed

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("Selected fraction (CV): " + result.fractionUsed());
        }
    }
}
```

```output
Selected fraction (CV): 0.3
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
