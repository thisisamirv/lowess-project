---
title: "OnlineLowess API"
weight: 34
---

For real-time data: processes one `(x, y)` point at a time and returns a smoothed value immediately once enough points have been seen.

See also: [API](api.md)

![Online Adapter](../assets/diagrams/online_comparison.svg)

## `fastlowess.DefaultOnlineOptions() OnlineOptions`

```go
opts := fastlowess.DefaultOnlineOptions()
opts.WindowCapacity = 200
opts.MinPoints = 10
```

`OnlineOptions` embeds [`Options`](api.md) (all the same fields apply, except `ReturnSE`, `CVFractions`/`CVMethod`/`CVK`/`CVSeed`, and `Backend`, which are batch-only). Note `Parallel` defaults to `false` for online use, since per-point updates rarely benefit from parallelism. Additional fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `WindowCapacity` | `int` | `1000` | Maximum number of recent points retained. |
| `MinPoints` | `int` | `3` | Minimum points required before output starts. |
| `UpdateMode` | `string` | `"full"` | How the window is updated as new points arrive. |

## `fastlowess.NewOnlineLowess(opts OnlineOptions) (*OnlineLowess, error)`

## `(*OnlineLowess) AddPoint(x, y float64) (res PointResult, ok bool, err error)`

Adds a single observation. `ok` is `false` while the window is still filling (fewer than `MinPoints` seen so far); once `ok` is `true`, `res` holds the smoothed value for the most recently added point.

## `(*OnlineLowess) Close() error`

Releases native resources. Safe to call multiple times.

## `PointResult` fields

| Field | Type | Notes |
| --- | --- | --- |
| `Y` | `float64` | Smoothed value. |
| `StandardError` | `float64` | `NaN` if not computed. |
| `Residual` | `float64` | `NaN` if not computed. |
| `RobustnessWeight` | `float64` | `NaN` if not computed. |
| `IterationsUsed` | `int` | `-1` if not applicable. |

## Example

```go
package main

import (
 "fmt"
 "log"
 "math"

 "github.com/thisisamirv/lowess-project/bindings/go/fastlowess"
)

func main() {
 const n = 100
 x := make([]float64, n)
 y := make([]float64, n)
 for i := 0; i < n; i++ {
  x[i] = float64(i) * 2 * math.Pi / float64(n-1)
  y[i] = math.Sin(x[i]) + 0.1
 }

 opts := fastlowess.DefaultOnlineOptions()
 opts.Fraction = 0.5
 opts.WindowCapacity = 50
 opts.MinPoints = 3

 model, err := fastlowess.NewOnlineLowess(opts)
 if err != nil {
  log.Fatal(err)
 }
 defer model.Close()

 _, ok1, err := model.AddPoint(x[0], y[0])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok1)

 _, ok2, err := model.AddPoint(x[1], y[1])
 if err != nil {
  log.Fatal(err)
 }
 fmt.Println(ok2)

 res, ok3, err := model.AddPoint(x[2], y[2])
 if err != nil {
  log.Fatal(err)
 }
 if ok3 {
  fmt.Println(res.Y)
 }
}
```

```output
false
false
0.22659245357374927
```
