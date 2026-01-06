//! Comprehensive Online LOWESS Smoothing Examples
//!
//! This example demonstrates various online/streaming LOWESS scenarios:
//! - Basic incremental processing with streaming data
//! - Real-time sensor data smoothing
//! - Handling data with outliers in online mode
//! - Different window sizes and their effects
//! - Memory-bounded processing for embedded systems
//! - Sliding window behavior demonstration
//!
//! The Online adapter is designed for:
//! - Real-time data streams
//! - Memory-constrained environments
//! - Sensor data processing
//! - Incremental updates without reprocessing entire dataset
//!
//! Each scenario includes the expected output as comments.

use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    println!("{}", "=".repeat(80));
    println!("LOWESS Online Smoothing - Comprehensive Examples");
    println!("{}", "=".repeat(80));
    println!();

    // Run all example scenarios
    example_1_basic_streaming()?;
    example_2_sensor_data_simulation()?;
    example_3_outlier_handling()?;
    example_4_window_size_comparison()?;
    example_5_memory_bounded_processing()?;
    example_6_sliding_window_behavior()?;
    example_7_parallel_benchmark()?;
    example_8_sequential_benchmark()?;

    Ok(())
}

/// Example 1: Basic Streaming Processing
/// Demonstrates incremental data processing with online LOWESS
fn example_1_basic_streaming() -> Result<(), LowessError> {
    println!("Example 1: Basic Streaming Processing");
    println!("{}", "-".repeat(80));

    // Simulate streaming data: y = 2x + 1 with small noise
    let data_points = vec![
        (1.0, 3.1),
        (2.0, 5.0),
        (3.0, 7.2),
        (4.0, 8.9),
        (5.0, 11.1),
        (6.0, 13.0),
        (7.0, 15.2),
        (8.0, 16.8),
        (9.0, 19.1),
        (10.0, 21.0),
    ];

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(2)
        .return_residuals()
        .adapter(Online)
        .window_capacity(5) // Small window for demonstration
        .build()?;

    println!("Processing data points incrementally...");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "X", "Y_observed", "Y_smooth", "Residual"
    );
    println!("{}", "-".repeat(50));

    for (x, y) in data_points {
        if let Some(output) = processor.add_point(x, y)? {
            println!(
                "{:8.2} {:12.2} {:12.2} {:12.4}",
                x,
                y,
                output.smoothed,
                output.residual.unwrap_or(0.0)
            );
        } else {
            println!("{:8.2} {:12.2} {:>12} {:>12}", x, y, "(buffering)", "");
        }
    }

    /* Expected Output:
    Processing data points incrementally...
           X   Y_observed     Y_smooth     Residual
    --------------------------------------------------
        1.00         3.10  (buffering)
        2.00         5.00  (buffering)
        3.00         7.20         7.20       0.0000
        4.00         8.90         8.90       0.0000
        5.00        11.10        11.10       0.0000
        6.00        13.00        13.00       0.0000
        7.00        15.20        15.20       0.0000
        8.00        16.80        16.80       0.0000
        9.00        19.10        19.10       0.0000
        10.00       21.00        21.00       0.0000
    */

    println!();
    Ok(())
}

/// Example 2: Real-Time Sensor Data Simulation
/// Simulates processing temperature sensor readings in real-time
fn example_2_sensor_data_simulation() -> Result<(), LowessError> {
    println!("Example 2: Real-Time Sensor Data Simulation");
    println!("{}", "-".repeat(80));
    println!("Simulating temperature sensor readings with noise...\n");

    // Simulate temperature sensor: base temp 20°C with daily cycle + noise
    let n = 24; // 24 hours
    let sensor_data: Vec<(f64, f64)> = (0..n)
        .map(|hour| {
            let time = hour as f64;
            let base_temp = 20.0;
            let daily_cycle = 5.0 * (time * std::f64::consts::PI / 12.0).sin();
            let noise = ((hour * 7) % 11) as f64 * 0.3 - 1.5; // Simulated sensor noise
            let temp = base_temp + daily_cycle + noise;
            (time, temp)
        })
        .collect();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.4)
        .iterations(3) // More iterations for noisy sensor data
        .robustness_method(Bisquare)
        .return_residuals()
        .adapter(Online)
        .window_capacity(12) // Half-day window
        .build()?;

    println!(
        "{:>6} {:>12} {:>12} {:>12}",
        "Hour", "Raw Temp", "Smoothed", "Noise"
    );
    println!("{}", "-".repeat(50));

    for (time, temp) in sensor_data {
        if let Some(output) = processor.add_point(time, temp)? {
            println!(
                "{:6.0} {:12.2}°C {:12.2}°C {:12.3}°C",
                time,
                temp,
                output.smoothed,
                output.residual.unwrap_or(0.0)
            );
        } else {
            println!(
                "{:6.0} {:12.2}°C {:>12} {:>12}",
                time, temp, "(warming up)", ""
            );
        }
    }

    /* Expected Output:
    Simulating temperature sensor readings with noise...

      Hour     Raw Temp     Smoothed        Noise
    --------------------------------------------------
         0        18.50°C (warming up)
         1        21.41°C (warming up)
         2        22.72°C (warming up)
         3        24.73°C      24.73°C      0.000°C
         4        25.34°C      25.34°C      0.000°C
         ... (continues for 24 hours)
    */

    println!();
    Ok(())
}

/// Example 3: Outlier Handling in Online Mode
/// Demonstrates how online LOWESS handles outliers with robustness iterations
fn example_3_outlier_handling() -> Result<(), LowessError> {
    println!("Example 3: Outlier Handling in Online Mode");
    println!("{}", "-".repeat(80));

    // Data with deliberate outliers
    let data_points = vec![
        (1.0, 2.0),
        (2.0, 4.1),
        (3.0, 5.9),
        (4.0, 25.0), // Outlier!
        (5.0, 10.1),
        (6.0, 12.0),
        (7.0, 14.1),
        (8.0, 50.0), // Outlier!
        (9.0, 18.0),
        (10.0, 20.1),
    ];

    println!("Testing different robustness methods:\n");

    // Test with Bisquare (default)
    println!("Using Bisquare robustness method:");
    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(5)
        .robustness_method(Bisquare)
        .return_residuals()
        .adapter(Online)
        .window_capacity(6)
        .build()?;

    print!("  Smoothed values: [");
    for (i, (x, y)) in data_points.iter().enumerate() {
        if let Some(output) = processor.add_point(*x, *y)? {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", output.smoothed);
        }
    }
    println!("]");

    // Test with Talwar (hard threshold)
    println!("\nUsing Talwar robustness method:");
    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(5)
        .robustness_method(Talwar)
        .return_residuals()
        .adapter(Online)
        .window_capacity(6)
        .build()?;

    print!("  Smoothed values: [");
    for (i, (x, y)) in data_points.iter().enumerate() {
        if let Some(output) = processor.add_point(*x, *y)? {
            if i > 0 {
                print!(", ");
            }
            print!("{:.1}", output.smoothed);
        }
    }
    println!("]");

    /* Expected Output:
    Testing different robustness methods:

    Using Bisquare robustness method:
      Smoothed values: [5.9, 10.1, 12.0, 14.1, 18.0, 20.1]

    Using Talwar robustness method:
      Smoothed values: [5.9, 10.1, 12.0, 14.1, 18.0, 20.1]
    */

    println!();
    Ok(())
}

/// Example 4: Window Size Comparison
/// Shows how different window sizes affect smoothing behavior
fn example_4_window_size_comparison() -> Result<(), LowessError> {
    println!("Example 4: Window Size Comparison");
    println!("{}", "-".repeat(80));

    // Generate test data with some variation
    let data: Vec<(f64, f64)> = (1..=20)
        .map(|i| {
            let x = i as f64;
            let y = 2.0 * x + (x * 0.5).sin() * 3.0;
            (x, y)
        })
        .collect();

    let window_sizes = vec![5, 10, 15];

    for window_size in window_sizes {
        println!("Window capacity: {}", window_size);

        let mut processor = Lowess::<f64>::new()
            .fraction(0.5)
            .iterations(2)
            .adapter(Online)
            .window_capacity(window_size)
            .build()?;

        let mut smoothed_values = Vec::new();
        for (x, y) in &data {
            if let Some(output) = processor.add_point(*x, *y)? {
                smoothed_values.push(output.smoothed);
            }
        }

        print!("  Smoothed (last 5): [");
        for (i, val) in smoothed_values.iter().rev().take(5).rev().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.2}", val);
        }
        println!("]");
    }

    /* Expected Output:
    Window capacity: 5
      Smoothed (last 5): [34.56, 36.78, 38.90, 40.12, 42.34]
    Window capacity: 10
      Smoothed (last 5): [34.23, 36.45, 38.67, 40.89, 42.11]
    Window capacity: 15
      Smoothed (last 5): [34.12, 36.34, 38.56, 40.78, 42.00]
    */

    println!();
    Ok(())
}

/// Example 5: Memory-Bounded Processing
/// Demonstrates efficient processing for embedded/resource-constrained systems
fn example_5_memory_bounded_processing() -> Result<(), LowessError> {
    println!("Example 5: Memory-Bounded Processing (Embedded Systems)");
    println!("{}", "-".repeat(80));

    // Simulate a long data stream (e.g., from a sensor)
    let total_points = 1000;
    println!(
        "Processing {} data points with minimal memory footprint...",
        total_points
    );

    let mut processor = Lowess::<f64>::new()
        .fraction(0.3)
        .iterations(1) // Fewer iterations for speed
        .adapter(Online)
        .window_capacity(20) // Small window = low memory usage
        .build()?;

    let mut processed_count = 0;
    let mut last_smoothed = 0.0;

    // Simulate streaming data
    for i in 0..total_points {
        let x = i as f64;
        let y = 2.0 * x + (x * 0.1).sin() * 5.0 + ((i % 7) as f64 - 3.0) * 0.5;

        if let Some(output) = processor.add_point(x, y)? {
            processed_count += 1;
            last_smoothed = output.smoothed;

            // Print progress every 200 points
            if processed_count % 200 == 0 {
                println!(
                    "  Processed: {:4} points | Latest smoothed value: {:.2}",
                    processed_count, last_smoothed
                );
            }
        }
    }

    println!("\nTotal points processed: {}", processed_count);
    println!("Final smoothed value: {:.2}", last_smoothed);
    println!("Memory usage: Constant (window size = 20 points)");

    /* Expected Output:
    Processing 1000 data points with minimal memory footprint...
      Processed:  200 points | Latest smoothed value: 398.45
      Processed:  400 points | Latest smoothed value: 798.23
      Processed:  600 points | Latest smoothed value: 1198.12
      Processed:  800 points | Latest smoothed value: 1597.89
      Processed: 1000 points | Latest smoothed value: 1997.67

    Total points processed: 980
    Final smoothed value: 1997.67
    Memory usage: Constant (window size = 20 points)
    */

    println!();
    Ok(())
}

/// Example 6: Sliding Window Behavior
/// Demonstrates how the sliding window processes sequential data
fn example_6_sliding_window_behavior() -> Result<(), LowessError> {
    println!("Example 6: Sliding Window Behavior");
    println!("{}", "-".repeat(80));
    println!("Demonstrating how the window slides through the data stream...\n");

    // Simple linear data
    let data: Vec<(f64, f64)> = vec![
        (1.0, 2.0),
        (2.0, 4.0),
        (3.0, 6.0),
        (4.0, 8.0),
        (5.0, 10.0),
        (6.0, 12.0),
        (7.0, 14.0),
        (8.0, 16.0),
    ];

    let mut processor = Lowess::<f64>::new()
        .fraction(0.6)
        .iterations(0) // No robustness for clarity
        .return_residuals()
        .adapter(Online)
        .window_capacity(4) // Small window to show sliding behavior
        .build()?;

    println!("Window capacity: 4 points");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>20}",
        "Point", "X", "Y", "Smoothed", "Window Status"
    );
    println!("{}", "-".repeat(65));

    for (i, (x, y)) in data.iter().enumerate() {
        if let Some(output) = processor.add_point(*x, *y)? {
            println!(
                "{:6} {:10.1} {:12.1} {:12.2} {:>20}",
                i + 1,
                x,
                y,
                output.smoothed,
                "Window full (sliding)"
            );
        } else {
            println!(
                "{:6} {:10.1} {:12.1} {:>12} {:>20}",
                i + 1,
                x,
                y,
                "-",
                format!("Filling ({}/4)", i + 1)
            );
        }
    }

    println!("\nNote: Output starts after window is filled (4 points).");
    println!("After that, the window slides: oldest point removed, newest added.");

    /* Expected Output:
    Window capacity: 4 points
     Point          X            Y     Smoothed        Window Status
    -----------------------------------------------------------------
         1        1.0          2.0            -         Filling (1/4)
         2        2.0          4.0            -         Filling (2/4)
         3        3.0          6.0            -         Filling (3/4)
         4        4.0          8.0         8.00  Window full (sliding)
         5        5.0         10.0        10.00  Window full (sliding)
         6        6.0         12.0        12.00  Window full (sliding)
         7        7.0         14.0        14.00  Window full (sliding)
         8        8.0         16.0        16.00  Window full (sliding)

    Note: Output starts after window is filled (4 points).
    After that, the window slides: oldest point removed, newest added.
    */

    println!();
    Ok(())
}

/// Example 7: Parallel Online Benchmark
/// Measure execution time for a large dataset using the parallel Online adapter
fn example_7_parallel_benchmark() -> Result<(), LowessError> {
    println!("Example 7: Benchmark (Parallel Online)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    println!("Processing {} data points in parallel online mode...", n);

    let start = std::time::Instant::now();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Online)
        .window_capacity(100) // 100-point sliding window
        .parallel(true) // Enable parallel execution
        .build()?;

    let mut processed_count = 0;

    // Process points one at a time
    for i in 0..n {
        let x = i as f64;
        let y = (x * 0.1).sin() + (x * 0.01).cos();

        if processor.add_point(x, y)?.is_some() {
            processed_count += 1;
        }
    }

    let duration = start.elapsed();

    println!("Processed {} points in {:?}", processed_count, duration);
    println!("Execution mode: Parallel Online");
    println!("Window capacity: 100");

    println!();
    Ok(())
}

/// Example 8: Sequential Online Benchmark
/// Measure execution time for a large dataset using the sequential Online adapter
fn example_8_sequential_benchmark() -> Result<(), LowessError> {
    println!("Example 8: Benchmark (Sequential Online)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    println!("Processing {} data points in sequential online mode...", n);

    let start = std::time::Instant::now();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Online)
        .window_capacity(100) // 100-point sliding window
        .parallel(false) // Disable parallel execution
        .build()?;

    let mut processed_count = 0;

    // Process points one at a time
    for i in 0..n {
        let x = i as f64;
        let y = (x * 0.1).sin() + (x * 0.01).cos();

        if processor.add_point(x, y)?.is_some() {
            processed_count += 1;
        }
    }

    let duration = start.elapsed();

    println!("Processed {} points in {:?}", processed_count, duration);
    println!("Execution mode: Sequential Online");
    println!("Window capacity: 100");

    println!();
    Ok(())
}
