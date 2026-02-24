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
    times <- 1:100
    temperatures <- 20 + 5 * sin(times / 10) + rnorm(100)

    model <- OnlineLowess(
        fraction = 0.3,
        window_capacity = 25,
        min_points = 5,
        update_mode = "incremental"
    )
    result <- model$add_points(times, temperatures)
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
    result = fl.smooth_online(
        times, temperatures,
        fraction=0.3,
        window_capacity=25,    # Keep last 25 points
        min_points=5,          # Wait for 5 points before output
        update_mode="incremental"
    )

    # Result contains smoothed values for each valid point
    print("Smoothed temperatures:", result["y"])
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    let mut processor = Lowess::new()
        .fraction(0.3)
        .iterations(1)
        .adapter(Online)
        .window_capacity(25)
        .min_points(5)
        .update_mode(Incremental)
        .build()?;

    // Simulate real-time data arrival
    for i in 0..100 {
        let x = i as f64;
        let y = 20.0 + 5.0 * (x / 10.0).sin() + rand::random::<f64>();
        
        if let Some(output) = processor.add_point(x, y)? {
            println!("Time {}: smoothed = {:.2}", x, output.smoothed);
        }
    }
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # Simulate sensor readings 
    times = collect(Float64, 1:100)
    temperatures = 20.0 .+ 5.0 .* sin.(times ./ 10.0) .+ randn(100)

    # Process with online mode
    result = smooth_online(
        times, temperatures,
        fraction=0.3,
        window_capacity=25,
        min_points=5,
        update_mode="incremental"
    )

    println("Smoothed temperatures: ", result.y)
    ```

=== "Node.js"
    ```javascript
    const { OnlineLowess } = require('fastlowess');

    const processor = new OnlineLowess(
        { fraction: 0.3, iterations: 1 },
        { windowCapacity: 25, minPoints: 5, updateMode: "incremental" }
    );

    // Simulate real-time data arrival
    for (let i = 0; i < 100; i++) {
        const x = i;
        const y = 20 + 5 * Math.sin(x / 10) + Math.random();
        
        const smoothed = processor.update(x, y);
        if (smoothed !== null) {
            console.log(`Time ${x}: smoothed = ${smoothed.toFixed(2)}`);
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    import { OnlineLowessWasm } from 'fastlowess-wasm';

    const processor = new OnlineLowessWasm(
        { fraction: 0.3, iterations: 1 },
        { windowCapacity: 25, minPoints: 5, updateMode: "incremental" }
    );

    // Simulate real-time data arrival
    for (let i = 0; i < readings.length; i++) {
        const smoothed = processor.update(readings[i].x, readings[i].y);
        if (smoothed !== null) {
            // Update dashboard UI
        }
    }
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    // Online mode processes points incrementally
    fastlowess::OnlineOptions opts;
    opts.fraction = 0.3;
    opts.iterations = 1;
    opts.window_capacity = 25;
    opts.min_points = 5;
    opts.update_mode = "incremental";

    auto result = fastlowess::online(times, temperatures, opts);

    // Result contains smoothed values
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << "Time " << result.x(i) << ": " << result.y(i) << std::endl;
    }
    ```

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

### Log File Processing

=== "R"
    ```r
    x <- seq(0, 100000, by = 1)
    y <- sin(x / 1000) + rnorm(length(x), sd = 0.1)

    model <- StreamingLowess(
        fraction = 0.05,
        chunk_size = 10000,
        overlap = 1000,
        merge_strategy = "weighted"
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
    
    result = fl.smooth_streaming(
        x, y,
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted"
    )
    
    print(f"Processed {len(result['y'])} points")
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;

    let mut processor = Lowess::new()
        .fraction(0.1)
        .iterations(2)
        .adapter(Streaming)
        .chunk_size(5000)
        .overlap(500)
        .merge_strategy(Weighted)
        .build()?;

    // Process chunks as they arrive
    let result1 = processor.process_chunk(&chunk1_x, &chunk1_y)?;
    let result2 = processor.process_chunk(&chunk2_x, &chunk2_y)?;

    // CRITICAL: Get buffered overlap data
    let final_result = processor.finalize()?;
    ```

=== "Julia"
    ```julia
    using FastLOWESS

    # Large dataset
    x = collect(range(0, 100000, step=1))
    y = sin.(x ./ 1000) .+ randn(length(x)) .* 0.1

    # Streaming mode handles everything internally
    result = smooth_streaming(
        x, y,
        fraction=0.05,
        chunk_size=10000,
        overlap=1000,
        merge_strategy="weighted"
    )
    ```

=== "Node.js"
    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const processor = new StreamingLowess(
        { fraction: 0.1, iterations: 2 },
        { chunkSize: 5000, overlap: 500 }
    );

    // Process chunks
    const r1 = processor.processChunk(chunk1_x, chunk1_y);
    const r2 = processor.processChunk(chunk2_x, chunk2_y);

    // Always get buffered data
    const finalResult = processor.finalize();
    ```

=== "WebAssembly"
    ```javascript
    import { StreamingLowessWasm } from 'fastlowess-wasm';

    const processor = new StreamingLowessWasm(
        { fraction: 0.1, iterations: 2 },
        { chunkSize: 5000, overlap: 500 }
    );

    // Process chunks as they arrive
    const result1 = processor.processChunk(x1, y1);
    const result2 = processor.processChunk(x2, y2);
    const finalResult = processor.finalize();
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    fastlowess::StreamingOptions opts;
    opts.fraction = 0.1;
    opts.iterations = 2;
    opts.chunk_size = 5000;
    opts.overlap = 500;

    auto result = fastlowess::streaming(x, y, opts);

    std::cout << "Processed " << result.size() << " points" << std::endl;
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
            result <- Lowess(fraction = 0.4)$fit(data_x, data_y)
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
            result = fl.smooth(np.array(data_x), np.array(data_y), fraction=0.4)
            current_smoothed = result["y"][-1]
    ```

=== "Node.js"
    ```javascript
    const fl = require('fastlowess');

    const windowCapacity = 50;
    let dataX = [], dataY = [];

    for (let i = 0; i < 200; i++) {
        dataX.push(i);
        dataY.push(25.0 + 10 * Math.sin(i / 20) + Math.random() * 4 - 2);

        if (dataX.length > windowCapacity) {
            dataX.shift();
            dataY.shift();
        }

        if (dataX.length >= 5) {
            const xArr = new Float64Array(dataX);
            const yArr = new Float64Array(dataY);
            const result = fl.smooth(xArr, yArr, { fraction: 0.4 });
            const currentSmoothed = result.y[result.y.length - 1];
        }
    }
    ```

=== "WebAssembly"
    ```javascript
    import { smooth } from 'fastlowess-wasm';

    // Sliding window logic
    for (const point of stream) {
        windowX.push(point.x);
        windowY.push(point.y);
        
        if (windowX.length > 50) {
            windowX.shift();
            windowY.shift();
        }

        const result = smooth(new Float64Array(windowX), new Float64Array(windowY), { 
            fraction: 0.4 
        });
        const smoothed = result.y[result.y.length - 1];
    }
    ```

=== "C++"
    ```cpp
    #include "fastlowess.hpp"

    // Sliding window logic
    for (const auto& point : stream) {
        windowX.push_back(point.x);
        windowY.push_back(point.y);
        
        if (windowX.size() > 50) {
            windowX.erase(windowX.begin());
            windowY.erase(windowY.begin());
        }

        auto result = fastlowess::smooth(windowX, windowY, { .fraction = 0.4 });
        const auto smoothed = result.y.back();
    }
    ```

---

## Choosing Parameters

### Online Mode

| Parameter         | Guidance                                         |
|-------------------|--------------------------------------------------|
| `window_capacity` | Enough history for `fraction` to work            |
| `min_points`      | 2–5 typically; higher for stability              |
| `update_mode`     | `"incremental"` for speed, `"full"` for accuracy |

### Streaming Mode

| Parameter        | Guidance                                                  |
|------------------|-----------------------------------------------------------|
| `chunk_size`     | Balance memory vs. processing overhead                    |
| `overlap`        | 10–20% of chunk_size for smooth transitions               |
| `merge_strategy` | `"weighted"` for best quality, `"average"` for simplicity |

---

## Performance Considerations

| Mode          | Memory         | Latency      | Use Case            |
|---------------|----------------|--------------|---------------------|
| **Online**    | Fixed (window) | ~1ms/point   | Sensors, dashboards |
| **Streaming** | ~chunk_size    | ~100ms/chunk | Large files, ETL    |
| **Batch**     | Full dataset   | N/A          | Analysis, reports   |

---

## See Also

- [Execution Modes](../user-guide/adapters.md) — Detailed mode comparison
- [Time Series](time-series.md) — General time series analysis
