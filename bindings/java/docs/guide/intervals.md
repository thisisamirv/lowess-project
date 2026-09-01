---
title: "Intervals"
weight: 60
---

Confidence and prediction intervals for uncertainty quantification.

## Overview

![Confidence and Prediction Intervals](../assets/diagrams/intervals_comparison.svg)

> **Adapter support:** Confidence and prediction intervals are available in **Batch** (`Lowess`) mode only. `StreamingLowess` and `OnlineLowess` do not support intervals.

| Type | Represents | Width | Use |
| --- | --- | --- | --- |
| **Confidence** | Uncertainty in mean curve | Narrow | Where is the true trend? |
| **Prediction** | Uncertainty for new points | Wide | Where will new data fall? |

---

## Confidence Intervals

Estimate uncertainty in the smoothed curve itself.

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
                .fraction(0.5)
                .confidenceIntervals(0.95) // 95% CI
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);

            double[] lower = result.confidenceLower().orElseThrow();
            double[] upper = result.confidenceUpper().orElseThrow();
            for (int i = 0; i < 3; i++) {
                System.out.printf("x=%.2f: y=%.2f [%.2f, %.2f]%n",
                        result.x()[i], result.y()[i], lower[i], upper[i]);
            }
        }
    }
}
```

```output
x=0.00: y=0.33 [0.29, 0.37]
x=0.06: y=0.36 [0.32, 0.40]
x=0.13: y=0.39 [0.34, 0.43]
```

---

## Prediction Intervals

Estimate where new observations might fall.

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
                .fraction(0.5)
                .predictionIntervals(0.95) // 95% PI
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.printf("Prediction bounds: [%.2f, %.2f]%n",
                    result.predictionLower().orElseThrow()[0], result.predictionUpper().orElseThrow()[0]);
        }
    }
}
```

```output
Prediction bounds: [-0.04, 0.71]
```

---

## Both Intervals

Request both types simultaneously:

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
                .fraction(0.5)
                .confidenceIntervals(0.95)
                .predictionIntervals(0.95)
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.printf("First point 95%% CI: [%s, %s]%n",
                    result.confidenceLower().orElseThrow()[0], result.confidenceUpper().orElseThrow()[0]);
        }
    }
}
```

```output
First point 95% CI: [0.29412988645250643, 0.3746108343833247]
```

---

## Confidence Levels

Common levels and their z-values:

| Level | z-value | Interpretation |
| --- | --- | --- |
| 0.90 | 1.645 | 90% of intervals contain true value |
| 0.95 | 1.960 | 95% of intervals contain true value |
| 0.99 | 2.576 | 99% of intervals contain true value |

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

        // 99% confidence interval
        Options options = Options.builder().confidenceIntervals(0.99).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("First lower CI bound (99%): " + result.confidenceLower().orElseThrow()[0]);
        }
    }
}
```

```output
First lower CI bound (99%): 0.31714272639100155
```

---

## Standard Errors

Access standard errors directly (available when intervals are computed):

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

        Options options = Options.builder().returnSe(true).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            double[] se = result.standardErrors().orElseThrow();
            for (int i = 0; i < 3; i++) {
                System.out.printf("Point %d: SE = %.4f%n", i, se[i]);
            }
        }
    }
}
```

```output
Point 0: SE = 0.0254
Point 1: SE = 0.0269
Point 2: SE = 0.0283
```

---

## Availability

> **Batch Mode Only:** Confidence and prediction intervals are only available in **Batch** mode. `StreamingLowess` and `OnlineLowess` do not support intervals.

| Feature | Batch | Streaming | Online |
| --- | --- | --- | --- |
| Confidence intervals | ✓ | ✗ | ✗ |
| Prediction intervals | ✓ | ✗ | ✗ |
| Standard errors | ✓ | ✗ | ✗ |

All examples above use the same 100-point noisy sine wave (see [Weight Functions](../weighting/kernels.md#choosing-a-kernel) for the generation code).
