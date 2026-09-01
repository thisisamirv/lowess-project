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
package main

import (
 "fmt"
 "log"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 const n = 20
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i)
  y[i] = float64(i) + 0.1
 }

 opts := fastlowess.DefaultStreamingOptions()
 opts.ChunkSize = 10
 opts.Overlap = 2

 model, err := fastlowess.NewStreamingLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 if _, err := model.ProcessChunk(x[:10], y[:10]); err != nil {
  log.Fatal(err)
 }
 if _, err := model.ProcessChunk(x[10:], y[10:]); err != nil {
  log.Fatal(err)
 }

 result, err := model.Finalize()
 if err != nil {
  log.Fatal(err)
 }
 fmt.Printf("y[0]: %.4f\n", result.Y[0])
}
```

```output
y[0]: 17.6391
```
