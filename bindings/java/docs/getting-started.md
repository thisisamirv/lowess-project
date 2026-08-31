---
title: "Getting Started"
weight: 10
---

## Installation

Add the dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>com.thisisamirv</groupId>
    <artifactId>fastlowess</artifactId>
    <version>3.1.0</version>
</dependency>
```

This binding loads a native library via JNI. See [installation.md](installation.md) for details on providing the native `fastlowess_java` library.

## A first fit

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
            x[i] = i / 10.0;
            y[i] = Math.sin(x[i]) + 0.1 * Math.sin(i);
        }

        Options options = Options.builder()
                .fraction(0.2) // smaller fraction = less smoothing, more local detail
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, y);

            for (int i = 0; i < 5; i++) {
                System.out.printf("x=%.2f  y=%.3f  smoothed=%.3f%n", result.x()[i], y[i], result.y()[i]);
            }
        }
    }
}
```

```output
x=0.00  y=0.000  smoothed=0.166
x=0.10  y=0.184  smoothed=0.215
x=0.20  y=0.290  smoothed=0.270
x=0.30  y=0.310  smoothed=0.331
x=0.40  y=0.314  smoothed=0.395
```

`fraction` is the most important tuning parameter: it controls the size of the local neighbourhood used at each point.

| Range | Effect | Use case |
| --- | --- | --- |
| 0.1-0.3 | Fine detail | Rapidly changing signals |
| 0.3-0.5 | Balanced | General purpose |
| 0.5-0.7 | Heavy smoothing | Noisy data |
| 0.7-1.0 | Very smooth | Trend extraction |

## Choosing a model

- **`Lowess`** (batch): the whole dataset fits in memory and you want every feature (intervals, cross-validation, GPU). Start here.
- **`StreamingLowess`**: the dataset doesn't fit in memory or arrives in chunks.
- **`OnlineLowess`**: you need a smoothed value immediately as each point arrives (real-time).

See [api.md](api.md), [api-streaming.md](api-streaming.md), and [api-online.md](api-online.md) for the full reference of each.

## Handling outliers

LOWESS can robustly handle outliers through iterative reweighting:

```java
import fastlowess.Lowess;
import fastlowess.Options;
import fastlowess.Result;

public class Example {
    public static void main(String[] args) {
        double[] x = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
        double[] yWithOutlier = { 2.0, 4.0, 6.0, 50.0, 10.0, 12.0 }; // 50.0 is an outlier

        Options options = Options.builder()
                .fraction(0.7)
                .iterations(3)                   // more iterations for outliers
                .robustnessMethod("bisquare")    // default, smooth downweighting
                .returnRobustnessWeights(true)   // see which points were downweighted
                .build();

        try (Lowess model = new Lowess(options)) {
            Result result = model.fit(x, yWithOutlier);

            // Outliers will have low robustness weights
            double[] weights = result.robustnessWeights().orElseThrow();
            for (int i = 0; i < weights.length; i++) {
                if (weights[i] < 0.5) {
                    System.out.printf("Point %d is likely an outlier (weight: %.3f)%n", i, weights[i]);
                }
            }
        }
    }
}
```

```output
Point 3 is likely an outlier (weight: 0.000)
```

## Streaming mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap:

```java
import fastlowess.Result;
import fastlowess.StreamingLowess;
import fastlowess.StreamingOptions;

public class Example {
    public static void main(String[] args) {
        int n = 5000;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 10.0 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i] / Math.PI) * Math.exp(-x[i] / 30.0);
        }

        int chunkSize = 1000;
        StreamingOptions options = StreamingOptions.builder()
                .fraction(0.2)
                .chunkSize(chunkSize)
                .overlap(100)
                .build();

        try (StreamingLowess model = new StreamingLowess(options)) {
            for (int i = 0; i < n; i += chunkSize) {
                int end = Math.min(i + chunkSize, n);
                double[] xChunk = java.util.Arrays.copyOfRange(x, i, end);
                double[] yChunk = java.util.Arrays.copyOfRange(y, i, end);
                model.processChunk(xChunk, yChunk);
            }

            Result result = model.finish();
            System.out.println("Smoothed " + result.y().length + " points in the final chunk");
        }
    }
}
```

```output
Smoothed 100 points in the final chunk
```

## Next steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [API Reference](api.md) |
| Batch vs Streaming vs Online | [Execution Modes](adapter-choice.md) |
| Edge handling | [Boundary Handling](boundary.md) |
| Outlier handling in depth | [Robustness](robustness.md) |
| Kernel functions | [Weight Functions](kernels.md) |
| Residual scale estimation | [Scaling Methods](scaling.md) |
| Automated parameter selection | [Cross-Validation](cross-validation.md) |
| Per-observation weights | [Custom Weights](custom-weights.md) |
