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

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    # Simulate sensor readings arriving over time
    np.random.seed(42)
    n_readings = 100
    times = np.arange(n_readings)
    temperatures = 20 + 5 * np.sin(times / 10) + np.random.normal(0, 1, n_readings)

    # Process with online mode
    online = fl.OnlineLowess(
        fraction=0.3,
        window_capacity=25,    # Keep last 25 points
        min_points=5,          # Wait for 5 points before output
        update_mode="incremental"
    )
    for xi, yi in zip(times, temperatures):
        result = online.add_point(float(xi), float(yi))
        if result is not None:
            print(f"Time {xi:.0f}: smoothed = {result.y:.2f}")
    ```
=== "Node.js"
    ```javascript
    const { OnlineLowess } = require('fastlowess');

    const processor = new OnlineLowess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 25, min_points: 5, update_mode: "incremental" }
    );

    // Simulate real-time data arrival
    for (let i = 0; i < 100; i++) {
        const x = i;
        const y = 20 + 5 * Math.sin(x / 10) + Math.random();
        
        const res = processor.add_point(x, y);
        if (res !== null) {
            console.log(`Time ${x}: smoothed = ${res.y.toFixed(2)}`);
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    const { OnlineLowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new OnlineLowess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 25, min_points: 5, update_mode: "incremental" }
    );

    for (let i = 0; i < x.length; i++) {
        const res = processor.add_point(x[i], y[i]);
        if (res !== undefined && res !== null) {
            // Update dashboard UI with res.y
        }
    }
    ```

=== "C++"
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

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    # Simulate large dataset arriving in chunks
    total_points = 100000
    chunk_size = 10000
    
    # All at once with streaming handles chunking internally
    x = np.arange(total_points, dtype=float)
    y = np.sin(x / 1000) + np.random.normal(0, 0.1, total_points)
    
    model = fl.StreamingLowess(
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted_average"
    )
    model.process_chunk(x, y)
    result = model.finalize()
    
    print(f"Processed {len(result.y)} points")
    ```
=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const chunk1_x = Float64Array.from({ length: 50 }, (_, i) => i);
    const chunk1_y = Float64Array.from(chunk1_x, v => Math.sin(v * 0.1));
    const chunk2_x = Float64Array.from({ length: 50 }, (_, i) => i + 50);
    const chunk2_y = Float64Array.from(chunk2_x, v => Math.sin(v * 0.1));

    const processor = new StreamingLowess(
        { fraction: 0.1, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    // Process chunks
    const r1 = processor.process_chunk(chunk1_x, chunk1_y);
    const r2 = processor.process_chunk(chunk2_x, chunk2_y);

    // Always get buffered data
    const finalResult = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    const { StreamingLowess } = require('fastlowess-wasm');

    const n = 50;
    const x1 = Float64Array.from({ length: n }, (_, i) => i);
    const y1 = Float64Array.from(x1, xi => Math.sin(xi * 0.1) + 0.1);
    const x2 = Float64Array.from({ length: n }, (_, i) => n + i);
    const y2 = Float64Array.from(x2, xi => Math.sin(xi * 0.1) + 0.1);

    const processor = new StreamingLowess(
        { fraction: 0.1, iterations: 2 },
        { chunk_size: 5000, overlap: 500 }
    );

    // Process chunks as they arrive
    const result1 = processor.process_chunk(x1, y1);
    const result2 = processor.process_chunk(x2, y2);
    const finalResult = processor.finalize();
    ```

=== "C++"
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

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    # Simulated real-time dashboard sliding window
    window_capacity = 50
    data_x, data_y = [], []
    
    for i in range(200):
        x, y = i, 25.0 + 10 * np.sin(i / 20) + np.random.normal(0, 2)
        data_x.append(x)
        data_y.append(y)
        
        if len(data_x) > window_capacity:
            data_x = data_x[-window_capacity:]
            data_y = data_y[-window_capacity:]
        
        if len(data_x) >= 5:
            model = fl.Lowess(fraction=0.4)
            result = model.fit(np.array(data_x, dtype=float), np.array(data_y, dtype=float))
            current_smoothed = result.y[-1]
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

    const window_capacity = 50;
    let dataX = [], dataY = [];

    for (let i = 0; i < 200; i++) {
        dataX.push(i);
        dataY.push(25.0 + 10 * Math.sin(i / 20) + Math.random() * 4 - 2);

        if (dataX.length > window_capacity) {
            dataX.shift();
            dataY.shift();
        }

        if (dataX.length >= 5) {
            const xArr = new Float64Array(dataX);
            const yArr = new Float64Array(dataY);
            const model = new fl.Lowess({ fraction: 0.4 });
            const result = model.fit(xArr, yArr);
            const currentSmoothed = result.y[result.y.length - 1];
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    const { Lowess } = require('fastlowess-wasm');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    // Sliding window logic
    const windowX = [], windowY = [];
    for (let i = 0; i < x.length; i++) {
        windowX.push(x[i]);
        windowY.push(y[i]);

        if (windowX.length > 50) {
            windowX.shift();
            windowY.shift();
        }

        if (windowX.length < 2) continue;
        const model = new Lowess({ fraction: 0.4 });
        const result = model.fit(new Float64Array(windowX), new Float64Array(windowY));
        const smoothed = result.y[result.y.length - 1];
    }
    ```

=== "C++"
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
