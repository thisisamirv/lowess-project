---
title: "OnlineLowess API"
weight: 34
---

For real-time data: processes one `(x, y)` point at a time and returns a smoothed value immediately once enough points have been seen.

See also: [API](api.md)

![Online Adapter](../assets/diagrams/online_comparison.svg)

## `OnlineOptions.builder() OnlineOptions.Builder`

```java
OnlineOptions options = OnlineOptions.builder()
        .windowCapacity(200)
        .minPoints(10)
        .build();
```

`OnlineOptions.Builder` exposes all the same settings as [`Options.Builder`](api.md) (except `returnSe`, `cvFractions`/`cvMethod`/`cvK`/`cvSeed`, and `backend`, which are batch-only). Note `parallel` still defaults to `true` in the builder, but per-point updates rarely benefit from parallelism in practice. Additional settings:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `windowCapacity` | `int` | `1000` | Maximum number of recent points retained. |
| `minPoints` | `int` | `2` | Minimum points required before output starts. |
| `updateMode` | `String` | `"incremental"` | How the window is updated as new points arrive: `incremental`, `full`. |

## `new OnlineLowess(OnlineOptions options)`

## `OnlineLowess.addPoint(double x, double y) Optional<PointResult>`

Adds a single observation. Returns `Optional.empty()` while the window is still filling (fewer than `minPoints` seen so far); once populated, the `PointResult` holds the smoothed value for the most recently added point.

## `OnlineLowess.close()`

Releases native resources. Safe to call multiple times. Implements `AutoCloseable`.

## `PointResult` accessors

| Accessor | Type | Notes |
| --- | --- | --- |
| `y()` | `double` | Smoothed value. |
| `standardError()` | `OptionalDouble` | Empty if not computed. |
| `residual()` | `OptionalDouble` | Empty if not computed. |
| `robustnessWeight()` | `OptionalDouble` | Empty if not computed. |
| `iterationsUsed()` | `OptionalInt` | Empty if not applicable. |

## Example

```java
import fastlowess.OnlineLowess;
import fastlowess.OnlineOptions;
import fastlowess.PointResult;

import java.util.Optional;

public class Example {
    public static void main(String[] args) {
        final int n = 100;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i * 2 * Math.PI / (n - 1);
            y[i] = Math.sin(x[i]) + 0.1;
        }

        OnlineOptions options = OnlineOptions.builder()
                .fraction(0.5)
                .windowCapacity(50)
                .minPoints(3)
                .build();

        try (OnlineLowess model = new OnlineLowess(options)) {
            Optional<PointResult> r1 = model.addPoint(x[0], y[0]);
            System.out.println(r1.isPresent());

            Optional<PointResult> r2 = model.addPoint(x[1], y[1]);
            System.out.println(r2.isPresent());

            Optional<PointResult> r3 = model.addPoint(x[2], y[2]);
            r3.ifPresent(r -> System.out.println(r.y()));
        }
    }
}
```

```output
false
false
0.22659245357374927
```
