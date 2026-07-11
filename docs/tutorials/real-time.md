<!-- markdownlint-disable MD033 -->
# Real-Time Processing

Streaming and online LOWESS for live data.

## Overview

When data arrives continuously—from sensors, logs, or streaming pipelines—you need incremental smoothing that doesn't require reprocessing the entire dataset.

---

## Online Mode: Point-by-Point

For true real-time applications where each point must be processed immediately.

### Sensor Data Example

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    library(rfastlowess)

    set.seed(42)
    times <- 1:100
    temperatures <- 20 + 5 * sin(times / 10) + rnorm(100)

    model <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        min_points = 5,
        update_mode = "incremental"
    )
    for (i in seq_along(times)) {
        result <- model$add_point(times[i], temperatures[i])
        if (!is.null(result))
            cat(sprintf("Time %d: %.2f\n", times[i], result$smoothed))
    }
    ```

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
            print(f"Time {xi:.0f}: smoothed = {result.smoothed:.2f}")
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {

        let mut processor = OnlineLowess::new()
            .fraction(0.3)
            .iterations(1)
            .window_capacity(25)
            .min_points(5)
            .update_mode("incremental")
            .build()?;

        // Simulate real-time data arrival
        for i in 0..100 {
            let xi = i as f64;
            let yi = 20.0 + 5.0 * (xi / 10.0).sin() + (xi * 1.7).sin() * 0.5;

            if let Some(output) = processor.add_point(xi, yi)? {
                println!("Time {}: smoothed = {:.2}", xi, output.smoothed);
            }
        }

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS
    using Random, Statistics

    rng = MersenneTwister(42)
    x = collect(range(0, 2π, length=100))
    y = sin.(x) .+ randn(rng, 100) .* 0.3

    using FastLOWESS

    # Simulate sensor readings 
    times = collect(Float64, 1:100)
    temperatures = 20.0 .+ 5.0 .* sin.(times ./ 10.0) .+ randn(100)

    # Process with online mode
    model = OnlineLowess(;
        fraction=0.3,
        window_capacity=25,
        min_points=5,
        update_mode="incremental"
    )
    for i in eachindex(times)
        result = add_point(model, times[i], temperatures[i])
        if result !== nothing
            println("Time $(times[i]): smoothed = $(round(result.smoothed; digits=2))")
        end
    end
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
            console.log(`Time ${x}: smoothed = ${res.smoothed.toFixed(2)}`);
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    const { OnlineLowess } = require('./fastlowess_wasm.js');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const processor = new OnlineLowess(
        { fraction: 0.3, iterations: 1 },
        { window_capacity: 25, min_points: 5, update_mode: "incremental" }
    );

    for (let i = 0; i < x.length; i++) {
        const res = processor.add_point(x[i], y[i]);
        if (res !== undefined) {
            // Update dashboard UI with res.smoothed
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
                std::cout << "Time " << times[i] << ": " << res.smoothed() << std::endl;
            }
        }

        return 0;
    }
    ```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

### Log File Processing

=== "R"
    ```r
    library(rfastlowess)
    set.seed(42)
    x <- seq(0, 2 * pi, length.out = 100)
    y <- sin(x) + rnorm(100, sd = 0.3)

    x <- seq(0, 100000, by = 1)
    y <- sin(x / 1000) + rnorm(length(x), sd = 0.1)

    model <- StreamingLowess(
        fraction = 0.05,
        chunk_size = 10000,
        overlap = 1000,
        merge_strategy = "weighted_average"
    )
    result <- model$process_chunk(x, y)
    final <- model$finalize()
    ```

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

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    fn main() -> Result<(), LowessError> {
        let chunk1_x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let chunk1_y: Vec<f64> = chunk1_x.iter().map(|&xi| xi.sin() + 0.1).collect();
        let chunk2_x: Vec<f64> = (50..100).map(|i| i as f64).collect();
        let chunk2_y: Vec<f64> = chunk2_x.iter().map(|&xi| xi.sin() + 0.1).collect();

        let mut processor = StreamingLowess::new()
            .fraction(0.1)
            .iterations(2)
            .chunk_size(50)
            .overlap(10)
            .merge_strategy("weighted_average")
            .build()?;

        // Process chunks as they arrive
        processor.process_chunk(&chunk1_x, &chunk1_y)?;
        processor.process_chunk(&chunk2_x, &chunk2_y)?;

        // CRITICAL: Get buffered overlap data
        let final_result = processor.finalize()?;

        Ok(())
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # Large dataset
    x = collect(0.0:1.0:100000.0)
    y = sin.(x ./ 1000) .+ randn(length(x)) .* 0.1

    # Streaming mode handles everything internally
    model = StreamingLowess(;
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted_average"
    )
    process_chunk(model, x, y)
    result = finalize(model)
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
    const { StreamingLowess } = require('./fastlowess_wasm.js');

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

!!! warning "Always call finalize()"
    The streaming adapter buffers overlap data. Always call `finalize()` to retrieve the last chunk.

---

## Real-Time Dashboard Example

=== "R"
    ```r
    library(rfastlowess)

    # Simulated real-time dashboard
    window_capacity <- 50
    data_x <- numeric(0)
    data_y <- numeric(0)

    for (i in 1:200) {
        x <- i
        y <- 25.0 + 10 * sin(i / 20) + rnorm(1, sd = 2)
        
        data_x <- c(data_x, x)
        data_y <- c(data_y, y)
        
        if (length(data_x) > window_capacity) {
            data_x <- tail(data_x, window_capacity)
            data_y <- tail(data_y, window_capacity)
        }
        
        if (length(data_x) >= 5) {
            model <- Lowess(fraction = 0.4)
            result <- model$fit(data_x, data_y)
            current_smoothed <- tail(result$y, 1)
        }
    }
    ```

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
    const { Lowess } = require('./fastlowess_wasm.js');

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
