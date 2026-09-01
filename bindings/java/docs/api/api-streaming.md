---
title: "StreamingLowess API"
weight: 32
---

For datasets that don't fit in memory or arrive in chunks. Processes data incrementally, merging overlapping regions between chunks.

See also: [API](api.md)

## `StreamingOptions.builder() StreamingOptions.Builder`

```java
StreamingOptions options = StreamingOptions.builder()
        .chunkSize(2000)
        .overlap(200)
        .build();
```

`StreamingOptions.Builder` exposes all the same settings as [`Options.Builder`](api.md) (except `returnSe`, `cvFractions`/`cvMethod`/`cvK`/`cvSeed`, and `backend`, which are batch-only), plus:

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `chunkSize` | `int` | `5000` | Number of points processed per chunk. |
| `overlap` | `int` | library default | Points shared between consecutive chunks. Negative means "use the library default" (`500`). |
| `mergeStrategy` | `String` | `"weighted_average"` | How overlapping chunk results are combined. |

*See also: [Merge Strategies](../advanced/merge.md)*

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## `new StreamingLowess(StreamingOptions options)`

## `StreamingLowess.processChunk(double[] x, double[] y) Result`

Fits and returns the result for one chunk. Call repeatedly as chunks arrive.

## `StreamingLowess.finish() Result`

Flushes any buffered data and returns the final merged result. Call once after the last `processChunk`.

## `StreamingLowess.close()`

Releases native resources. Safe to call multiple times. Implements `AutoCloseable`.

## Example

```java
import fastlowess.Result;
import fastlowess.StreamingLowess;
import fastlowess.StreamingOptions;

public class Example {
    public static void main(String[] args) {
        final int n = 20;
        double[] x = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = i;
            y[i] = i + 0.1;
        }

        StreamingOptions options = StreamingOptions.builder()
                .chunkSize(10)
                .overlap(2)
                .build();

        try (StreamingLowess model = new StreamingLowess(options)) {
            double[] x1 = java.util.Arrays.copyOfRange(x, 0, 10);
            double[] y1 = java.util.Arrays.copyOfRange(y, 0, 10);
            double[] x2 = java.util.Arrays.copyOfRange(x, 10, 20);
            double[] y2 = java.util.Arrays.copyOfRange(y, 10, 20);
            model.processChunk(x1, y1);
            model.processChunk(x2, y2);

            Result result = model.finish();
            System.out.printf("y[0]: %.4f%n", result.y()[0]);
        }
    }
}
```

```output
y[0]: 17.6391
```
