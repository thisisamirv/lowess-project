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

- [Execution Modes](../user-guide/adapters.md) — Detailed mode comparison
- [Merge Strategies](../user-guide/merge.md) — Chunk reconciliation in depth
- [Scaling Methods](../user-guide/scaling.md) — Robustness scale estimation
- [Time Series](time-series.md) — General time series analysis
