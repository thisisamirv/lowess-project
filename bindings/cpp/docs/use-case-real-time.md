<!-- markdownlint-disable MD033 -->
# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

`window_capacity = 25` limits the internal buffer to the 25 most recent observations; each `add_point` call costs O(window) rather than growing with total history. `min_points = 5` suppresses output until the window holds enough points for a stable fit — calls made before that threshold return `null`/`None`/`nothing`. `update_mode = "incremental"` re-fits only the most recent point rather than the full window, halving typical latency at a modest accuracy cost.

### Sensor Data Example

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> times(n), temperatures(n);
    for (int i = 0; i < n; ++i) {
        times[i] = i * 0.1;
        temperatures[i] = 20.0 + std::sin(times[i]);
    }

    // Online mode processes points incrementally
    fastlowess::OnlineOptions opts;
    opts.fraction = 0.3;
    opts.iterations = 1;
    opts.window_capacity = 25;
    opts.min_points = 5;
    opts.update_mode = "incremental";

    fastlowess::OnlineLowess model(opts);
    for (size_t i = 0; i < times.size(); ++i) {
        auto res = model.add_point(times[i], temperatures[i]).value();
        if (res.has_value()) {
            std::cout << "Time " << times[i] << ": " << res.y() << std::endl;
        }
    }

    return 0;
}
```

```output
Time 0.4: 20.3894
Time 0.5: 20.4794
Time 0.6: 20.5646
Time 0.7: 20.6442
Time 0.8: 20.7174
Time 0.9: 20.7833
Time 1: 20.8424
Time 1.1: 20.8922
Time 1.2: 20.9331
Time 1.3: 20.9664
Time 1.4: 20.9884
Time 1.5: 21.0005
Time 1.6: 21.0053
Time 1.7: 20.9975
Time 1.8: 20.9796
Time 1.9: 20.952
Time 2: 20.9184
Time 2.1: 20.872
Time 2.2: 20.8169
Time 2.3: 20.7576
Time 2.4: 20.6867
Time 2.5: 20.6088
Time 2.6: 20.5249
Time 2.7: 20.4358
Time 2.8: 20.3422
Time 2.9: 20.2453
Time 3: 20.1459
Time 3.1: 20.0451
Time 3.2: 19.9438
Time 3.3: 19.843
Time 3.4: 19.7439
Time 3.5: 19.6473
Time 3.6: 19.5542
Time 3.7: 19.4656
Time 3.8: 19.3823
Time 3.9: 19.3052
Time 4: 19.235
Time 4.1: 19.1725
Time 4.2: 19.1182
Time 4.3: 19.0728
Time 4.4: 19.0366
Time 4.5: 19.01
Time 4.6: 18.9933
Time 4.7: 18.9867
Time 4.8: 18.9902
Time 4.9: 19.0038
Time 5: 19.0274
Time 5.1: 19.0607
Time 5.2: 19.1033
Time 5.3: 19.155
Time 5.4: 19.215
Time 5.5: 19.2829
Time 5.6: 19.358
Time 5.7: 19.4395
Time 5.8: 19.5266
Time 5.9: 19.6184
Time 6: 19.714
Time 6.1: 19.8125
Time 6.2: 19.9129
Time 6.3: 20.0141
Time 6.4: 20.1152
Time 6.5: 20.2151
Time 6.6: 20.3129
Time 6.7: 20.4076
Time 6.8: 20.4982
Time 6.9: 20.5838
Time 7: 20.6636
Time 7.1: 20.7367
Time 7.2: 20.8025
Time 7.3: 20.8603
Time 7.4: 20.9094
Time 7.5: 20.9495
Time 7.6: 20.9801
Time 7.7: 21.0009
Time 7.8: 21.0117
Time 7.9: 21.0124
Time 8: 21.003
Time 8.1: 20.9836
Time 8.2: 20.9543
Time 8.3: 20.9155
Time 8.4: 20.8676
Time 8.5: 20.8109
Time 8.6: 20.7462
Time 8.7: 20.674
Time 8.8: 20.5951
Time 8.9: 20.5103
Time 9: 20.4203
Time 9.1: 20.3262
Time 9.2: 20.2287
Time 9.3: 20.129
Time 9.4: 20.028
Time 9.5: 19.9268
Time 9.6: 19.8262
Time 9.7: 19.7274
Time 9.8: 19.6314
Time 9.9: 19.539
```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

`chunk_size` controls how many data points are processed in one pass; matching it to your file-read buffer or message-batch size avoids unnecessary copying. `overlap` retains that many points from the previous chunk as context so the neighbourhood at chunk boundaries is not artificially truncated. `merge_strategy = "weighted_average"` blends the overlapping region smoothly; use `"last"` if chunk boundaries are guaranteed to be well separated and no blending is needed.

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Call `finalize()` after the last chunk to retrieve the buffered tail.

### Log File Processing

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }


    fastlowess::StreamingOptions opts;
    opts.fraction = 0.1;
    opts.iterations = 2;
    opts.chunk_size = 5000;
    opts.overlap = 500;

    fastlowess::StreamingLowess stream(opts);
    (void)stream.process_chunk(x, y);
    auto result = stream.finalize().value();

    std::cout << "Processed " << result.y_vector().size() << " points" << std::endl;

    return 0;
}
```

```output
Processed 100 points
```

---

## Real-Time Dashboard Example

The dashboard pattern uses a plain LOWESS fit on a manually managed sliding window rather than `OnlineLowess`. This is the simplest approach when your UI framework already owns the data buffer and you only need the most recent smoothed value per frame. The trade-off is a full O(window²) refit on every tick; for high-frequency streams prefer `OnlineLowess` with `update_mode = "incremental"` to bound per-frame cost.

```cpp
#include <fastlowess.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    const int n = 100;
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = i * 2 * M_PI / (n - 1);
        y[i] = std::sin(x[i]) + 0.1;
    }

    std::vector<double> windowX, windowY;

    // Sliding window over preamble x/y data
    for (std::size_t i = 0; i < n; ++i) {
        windowX.push_back(x[i]);
        windowY.push_back(y[i]);

        if (windowX.size() > 50) {
            windowX.erase(windowX.begin());
            windowY.erase(windowY.begin());
        }

        if (windowX.size() < 2) continue;
        fastlowess::LowessOptions sw_opts;
        sw_opts.fraction = 0.4;
        fastlowess::Lowess model(sw_opts);
        auto result = model.fit(windowX, windowY).value();
        const auto smoothed = result.y_vector().back();
        (void)smoothed;
    }

    return 0;
}
```

---

## Choosing Parameters

### Online Mode

| Parameter | Guidance |
| --- | --- |
| `window_capacity` | Enough history for `fraction` to work |
| `min_points` | 2–5 typically; higher for stability |
| `update_mode` | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter | Guidance |
| --- | --- |
| `chunk_size` | Balance memory vs. processing overhead |
| `overlap` | 10–20% of chunk_size for smooth transitions |
| `merge_strategy` | `"weighted_average"` for best quality, `"average"` for simplicity |

---

## Performance Considerations

| Mode | Memory | Latency | Use Case |
| --- | --- | --- | --- |
| **Online** | Fixed (window) | ~1ms/point | Sensors, dashboards |
| **Streaming** | ~chunk_size | ~100ms/chunk | Large files, ETL |
| **Batch** | Full dataset | N/A | Analysis, reports |

---

## See Also

- [Execution Modes](adapter-choice.md) — Detailed mode comparison
- [Merge Strategies](merge.md) — Chunk reconciliation in depth
- [Scaling Methods](scaling.md) — Robustness scale estimation
- [Time Series](use-case-time-series.md) — General time series analysis
