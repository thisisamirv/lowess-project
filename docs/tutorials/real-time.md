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

    result <- fastlowess_online(
        times, temperatures,
        fraction = 0.3,
        window_capacity = 25,
        min_points = 5,
        update_mode = "incremental"
    )
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
    using fastLowess

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

---

## Streaming Mode: Chunk Processing

For large datasets that arrive in batches or files.

### Log File Processing

=== "R"
    ```r
    x <- seq(0, 100000, by = 1)
    y <- sin(x / 1000) + rnorm(length(x), sd = 0.1)

    result <- fastlowess_streaming(
        x, y,
        fraction = 0.05,
        chunk_size = 10000,
        overlap = 1000,
        merge_strategy = "weighted"
    )
    ```

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np

    # Simulate large dataset arriving in chunks
    total_points = 100000
    chunk_size = 10000
    
    all_smoothed = []
    
    for chunk_start in range(0, total_points, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_points)
        
        # Simulate chunk data
        x_chunk = np.arange(chunk_start, chunk_end, dtype=float)
        y_chunk = np.sin(x_chunk / 1000) + np.random.normal(0, 0.1, len(x_chunk))
        
        # If using the all-at-once API:
        # This will handle chunking internally
    
    # All at once with streaming
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
    let chunk1_result = processor.process_chunk(&chunk1_x, &chunk1_y)?;
    write_output(&chunk1_result.y);

    let chunk2_result = processor.process_chunk(&chunk2_x, &chunk2_y)?;
    write_output(&chunk2_result.y);

    // CRITICAL: Get buffered overlap data
    let final_result = processor.finalize()?;
    write_output(&final_result.y);
    ```

=== "Julia"
    ```julia
    using fastLowess

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

    println("Processed $(length(result.y)) points")
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
        # Simulate sensor reading
        x <- i
        y <- 25.0 + 10 * sin(i / 20) + rnorm(1, sd = 2)
        
        data_x <- c(data_x, x)
        data_y <- c(data_y, y)
        
        # Keep sliding window
        if (length(data_x) > window_capacity) {
            data_x <- tail(data_x, window_capacity)
            data_y <- tail(data_y, window_capacity)
        }
        
        # Smooth current window
        if (length(data_x) >= 5) {
            result <- fastlowess(data_x, data_y, fraction = 0.4)
            current_smoothed <- tail(result$y, 1)
            cat(sprintf("Reading %d: raw=%.1f, smoothed=%.1f\n", i, y, current_smoothed))
        }
        
        Sys.sleep(0.1)  # Simulate 10 Hz sampling
    }
    ```

=== "Python"
    ```python
    import fastlowess as fl
    import numpy as np
    import time

    # Simulated real-time dashboard
    window_capacity = 50
    data_x = []
    data_y = []
    
    for i in range(200):
        # Simulate sensor reading
        x = i
        y = 25.0 + 10 * np.sin(i / 20) + np.random.normal(0, 2)
        
        data_x.append(x)
        data_y.append(y)
        
        # Keep sliding window
        if len(data_x) > window_capacity:
            data_x = data_x[-window_capacity:]
            data_y = data_y[-window_capacity:]
        
        # Smooth current window
        if len(data_x) >= 5:
            result = fl.smooth(
                np.array(data_x), 
                np.array(data_y),
                fraction=0.4
            )
            current_smoothed = result["y"][-1]
            print(f"Reading {i}: raw={y:.1f}, smoothed={current_smoothed:.1f}")
        
        time.sleep(0.1)  # Simulate 10 Hz sampling
    ```

=== "Rust"
    ```rust
    use fastLowess::prelude::*;
    use std::{thread, time::Duration};

    let mut window_x: Vec<f64> = Vec::with_capacity(50);
    let mut window_y: Vec<f64> = Vec::with_capacity(50);

    for i in 0..200 {
        let x = i as f64;
        let y = 25.0 + 10.0 * (x / 20.0).sin() + rand::random::<f64>() * 4.0 - 2.0;
        
        window_x.push(x);
        window_y.push(y);
        
        // Keep sliding window
        if window_x.len() > 50 {
            window_x.remove(0);
            window_y.remove(0);
        }
        
        // Smooth current window
        if window_x.len() >= 5 {
            let model = Lowess::new()
                .fraction(0.4)
                .adapter(Batch)
                .build()?;
            let result = model.fit(&window_x, &window_y)?;
            let current_smoothed = result.y.last().unwrap();
            println!("Reading {}: raw={:.1}, smoothed={:.1}", i, y, current_smoothed);
        }
        
        thread::sleep(Duration::from_millis(100));
    }
    ```

=== "Julia"
    ```julia
    using fastLowess

    # Simulated real-time dashboard
    window_capacity = 50
    data_x = Float64[]
    data_y = Float64[]

    for i in 1:200
        # Simulate sensor reading
        x = Float64(i)
        y = 25.0 + 10.0 * sin(i / 20.0) + randn() * 2.0
        
        push!(data_x, x)
        push!(data_y, y)
        
        # Keep sliding window
        if length(data_x) > window_capacity
            popfirst!(data_x)
            popfirst!(data_y)
        end
        
        # Smooth current window
        if length(data_x) >= 5
            result = smooth(data_x, data_y, fraction=0.4)
            current_smoothed = last(result.y)
            @printf("Reading %d: raw=%.1f, smoothed=%.1f\n", i, y, current_smoothed)
        end
        
        sleep(0.1)  # Simulate 10 Hz sampling
    end
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

## Error Handling

=== "R"
    ```r
    tryCatch({
        result <- fastlowess_online(x, y, window_capacity = 10)
    }, error = function(e) {
        cat("Smoothing failed:", conditionMessage(e), "\n")
        # Fall back to raw data or last known good value
    })
    ```

=== "Python"
    ```python
    try:
        result = fl.smooth_online(x, y, window_capacity=10)
    except fl.LowessError as e:
        print(f"Smoothing failed: {e}")
        # Fall back to raw data or last known good value
    ```

=== "Rust"
    ```rust
    match processor.add_point(x, y) {
        Ok(Some(output)) => {
            println!("Smoothed: {:.2}", output.smoothed);
        }
        Ok(None) => {
            // Not enough points yet
        }
        Err(e) => {
            eprintln!("Smoothing failed: {}", e);
            // Fall back to raw data
        }
    }
    ```

=== "Julia"
    ```julia
    try
        result = smooth_online(x, y, window_capacity=10)
    catch e
        println("Smoothing failed: ", e)
        # Fall back to raw data or last known good value
    end
    ```

---

## See Also

- [Execution Modes](../user-guide/adapters.md) — Detailed mode comparison
- [Time Series](time-series.md) — General time series analysis
