---
title: "Custom Weights"
weight: 70
---

Per-observation weights that encode data quality directly into the LOWESS fit.

## How Custom Weights Work

Standard LOWESS assigns equal prior trust to all observations. Custom weights let you override this assumption point by point — before any distance or robustness weighting is applied.

The effective weight of observation $j$ in a local fit centred at $x_i$ is:

$$w_{ij} = \text{customWeights}_j \times K\!\left(\frac{d_{ij}}{h_i}\right) \times r_j$$

where $K$ is the distance kernel, $h_i$ is the local bandwidth, and $r_j$ is the robustness weight from the current iteration.

> **Batch adapter only:** `customWeights` applies in **Batch** (`Lowess`) mode. It is silently ignored in `StreamingLowess` and `OnlineLowess`.

---

## When to Use Custom Weights

| Situation | Recommended weight |
| --- | --- |
| Point known to be erroneous | `0.0` — fully excluded |
| Unreliable sensor / low precision | `0.1 – 0.5` |
| Standard observation | `1.0` (default) |
| Carefully calibrated measurement | `> 1.0` |
| Measurement uncertainty $\sigma_i$ | $1 / \sigma_i^2$ |

### Custom Weights vs. Robustness Iterations

Both mechanisms handle unreliable data, but they serve different purposes:

| | Custom Weights | Robustness Iterations |
| --- | --- | --- |
| **When known** | Before fitting | Computed from residuals |
| **Knowledge required** | Prior knowledge of quality | None — data-driven |
| **Effect** | Fixed throughout fit | Adapts each iteration |
| **Use case** | Known bad sensors, calibration | Unknown outlier contamination |

They compose: you can use both simultaneously. Custom weights suppress *a priori* bad points; robustness iterations then handle any *residual* outliers that remain.

---

## Basic Usage

### Suppress a Known Outlier

Set the weight to `0` at the bad point — it is excluded from every local fit that would otherwise include it.

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        double[] x = new double[10];
        double[] y = new double[10];
        for (int i = 0; i < 10; i++) {
            x[i] = i;
            y[i] = i * 2.0;
        }
        y[5] = 100.0; // spike

        double[] weights = new double[10];
        java.util.Arrays.fill(weights, 1.0);
        weights[5] = 0.0; // exclude the spike

        Options options = Options.builder().fraction(0.5).iterations(0).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y, weights);
            System.out.println("First smoothed value (custom weights): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (custom weights): 0.5726210350584308
```

---

### Emphasize Important Points

Assign high weights to measurements you trust most — calibration standards, reference instruments, or low-noise observations.

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

        double[] weights = new double[x.length];
        java.util.Arrays.fill(weights, 1.0);
        for (int i : new int[] { 5, 20, 40, 60, 80 }) {
            weights[i] = 10.0; // trust calibration 10x more
        }

        Options options = Options.builder().fraction(0.5).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y, weights);
            System.out.println("First smoothed value (custom weights): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (custom weights): 0.32374197253806136
```

---

### Propagate Measurement Uncertainty

If each observation has a known standard deviation $\sigma_i$, set $w_i = 1 / \sigma_i^2$ to give the fit information-theoretically optimal weighting.

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

        double[] sigma = new double[x.length];
        double[] weights = new double[x.length];
        for (int i = 0; i < sigma.length; i++) {
            sigma[i] = 0.1 + (i % 4) * 0.1;
            weights[i] = 1.0 / (sigma[i] * sigma[i]);
        }

        Options options = Options.builder().fraction(0.5).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y, weights);
            System.out.println("First smoothed value (custom weights): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (custom weights): 0.1522219803222456
```

---

## Combined with Robustness Iterations

Custom weights and robustness iterations compose naturally: use custom weights for *known* bad points and robustness for *unknown* contamination.

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        double[] x2 = new double[20];
        double[] y2 = new double[20];
        for (int i = 0; i < 20; i++) {
            x2[i] = i;
            y2[i] = i * 1.5;
        }
        y2[3] = -50.0; // known bad
        y2[12] = 80.0; // unknown outlier

        double[] weights2 = new double[20];
        java.util.Arrays.fill(weights2, 1.0);
        weights2[3] = 0.0;

        Options options = Options.builder().fraction(0.4).iterations(3).build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x2, y2, weights2);
            System.out.println("First smoothed value (custom weights): " + result.y()[0]);
        }
    }
}
```

```output
First smoothed value (custom weights): 1.7983966900445223
```

---

## Validation Rules

| Rule | Effect |
| --- | --- |
| Length must equal `n` | `RuntimeException` at `fit` time if mismatched |
| All values must be ≥ 0 | Negative weights are rejected |
| All-zero weight vector | Error: no points remain for any local fit |
| Uniform weights (`1.0` everywhere) | Identical result to omitting weights |

> **Zero-weight windows:** If a local neighbourhood contains only zero-weight points, the fit at that centre point falls back to the behaviour specified by `zeroWeightFallback` (default: `"use_local_mean"`).

---

## See Also

- [Robustness](robustness.md) — adaptive outlier downweighting via IRLS
- [API Reference](api.md) — full parameter reference
