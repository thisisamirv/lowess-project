---
title: "Scaling Methods"
weight: 50
---

Residual scale estimation during robustness iterations.

## Overview

When `iterations > 0`, LOWESS computes robustness weights by comparing each residual to the current residual scale estimate. `scalingMethod` controls how that scale is measured.

The robustness weight for point $i$ is:

$$w_i = B\!\left(\frac{|r_i|}{6 \cdot \hat{\sigma}}\right)$$

where $B$ is the bisquare function and $\hat{\sigma}$ is the scale estimate. A larger $\hat{\sigma}$ makes the algorithm more tolerant of large residuals; a smaller one makes it more aggressive.

| Method | Formula | Robustness | Speed |
| --- | --- | --- | --- |
| `"mad"` | Median of \|residuals − median(residuals)\| | Very robust | Moderate |
| `"mar"` | Median of \|residuals\| | Robust | Fast |
| `"mean"` | Mean of \|residuals\| | Less robust | Fastest |

![Scaling Methods Comparison](../assets/diagrams/scaling_comparison.svg)

---

## MAD — Median Absolute Deviation (Default)

$$\hat{\sigma} = \text{median}(|r_i - \text{median}(r_i)|)$$

First centers residuals at their median, then takes the median of the absolute deviations. Double use of the median makes it highly resistant to extreme outliers. This is the standard choice for robust regression.

**Use when**: Data may contain outliers (default for most applications).

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
                .iterations(3)
                .scalingMethod("mad")
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("First smoothed value (mad scaling): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (mad scaling): 0.38260776436644134
```

---

## MAR — Median Absolute Residual

$$\hat{\sigma} = \text{median}(|r_i|)$$

Uses the uncentered median — unlike MAD it does not subtract the residual median first. Still robust (median-based) but slightly less resistant than MAD when residuals are systematically shifted. Faster than MAD in practice because it requires only one partial sort.

**Use when**: Speed matters and data have minimal systematic bias in residuals.

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
                .iterations(3)
                .scalingMethod("mar")
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("First smoothed value (mar scaling): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (mar scaling): 0.38260776436644134
```

---

## Mean — Mean Absolute Residual

$$\hat{\sigma} = \frac{1}{n}\sum_i |r_i|$$

Arithmetic mean of absolute residuals. Non-robust: a single extreme outlier inflates $\hat{\sigma}$, causing the algorithm to under-downweight it. Fastest to compute (no sort required). Useful when data are believed to be clean and speed is a priority.

**Use when**: Clean data with no outliers; maximum computation speed required.

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
                .iterations(3)
                .scalingMethod("mean")
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);
            System.out.println("First smoothed value (mean scaling): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (mean scaling): 0.3821886150146138
```

---

## Choosing a Scaling Method

| Situation | Recommended Method |
| --- | --- |
| General purpose, possible outliers | `"mad"` (default) |
| Speed matters; residuals have minimal systematic bias | `"mar"` |
| Clean data, no outliers | `"mean"` |

See [Robustness](robustness.md) for a broader discussion of outlier handling.
