---
title: "StreamingLowess API"
weight: 32
---

For datasets that don't fit in memory or arrive in chunks. Processes data incrementally, merging overlapping regions between chunks.

See also: [API](api.md)

## `fastlowess.DefaultStreamingOptions() StreamingOptions`

```go
opts := fastlowess.DefaultStreamingOptions()
opts.ChunkSize = 2000
opts.Overlap = 200
```

`StreamingOptions` embeds [`Options`](api.md) (all the same fields apply, except `ReturnSE`, `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, and `Backend`, which are batch-only), plus:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `ChunkSize` | `int` | `5000` | Number of points processed per chunk. |
| `Overlap` | `int` | library default | Points shared between consecutive chunks. Negative means "use the library default". |
| `MergeStrategy` | `string` | `"weighted_average"` | How overlapping chunk results are combined. |

*See also: [Merge Strategies](../advanced/merge.md)*

![Merge Strategies](../assets/diagrams/merge_comparison.svg)

## `fastlowess.NewStreamingLowess(opts StreamingOptions) (*StreamingLowess, error)`

## `(*StreamingLowess) ProcessChunk(x, y []float64) (Result, error)`

Fits and returns the result for one chunk. Call repeatedly as chunks arrive.

## `(*StreamingLowess) Finalize() (Result, error)`

Flushes any buffered data and returns the final merged result. Call once after the last `ProcessChunk`.

## `(*StreamingLowess) Close() error`

Releases native resources. Safe to call multiple times.

## Example

```go
opts := fastlowess.DefaultStreamingOptions()
opts.ChunkSize = 1000

model, err := fastlowess.NewStreamingLowess(opts)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

for _, chunk := range chunks {
    if _, err := model.ProcessChunk(chunk.X, chunk.Y); err != nil {
        log.Fatal(err)
    }
}

result, err := model.Finalize()
if err != nil {
    log.Fatal(err)
}
fmt.Println(len(result.Y))
```
