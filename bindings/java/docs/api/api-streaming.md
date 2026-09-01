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
| `chunkSize` | `int` | `1000` | Number of points processed per chunk. |
| `overlap` | `int` | `0` | Points shared between consecutive chunks. |
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

StreamingOptions options = StreamingOptions.builder().chunkSize(1000).build();

try (StreamingLowess model = new StreamingLowess(options)) {
    for (Chunk chunk : chunks) {
        model.processChunk(chunk.x(), chunk.y());
    }

    Result result = model.finish();
    System.out.println(result.y().length);
}
```
