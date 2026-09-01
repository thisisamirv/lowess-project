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
opts := fastlowess.DefaultOnlineOptions()
model, err := fastlowess.NewOnlineLowess(opts)
if err != nil {
    log.Fatal(err)
}
defer model.Close()

for point := range sensorReadings {
    res, ok, err := model.AddPoint(point.X, point.Y)
    if err != nil {
        log.Fatal(err)
    }
    if ok {
        fmt.Println("smoothed:", res.Y)
    }
}
```
