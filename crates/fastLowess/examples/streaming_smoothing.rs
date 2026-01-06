//! Comprehensive Streaming LOWESS Smoothing Examples
//!
//! This example demonstrates various streaming/chunked LOWESS scenarios:
//! - Basic chunked processing for large datasets
//! - Different chunk sizes and overlap strategies
//! - Processing very large datasets that don't fit in memory
//! - Handling data with outliers in streaming mode
//! - Merge strategies for chunk boundaries
//! - File-based streaming simulation
//! - Performance comparison with different configurations
//!
//! The Streaming adapter is designed for:
//! - Large datasets (>100K points) that don't fit in memory
//! - Batch processing pipelines
//! - File-based data processing
//! - ETL (Extract, Transform, Load) workflows
//!
//! Each scenario includes the expected output as comments.

use fastLowess::prelude::*;

fn main() -> Result<(), LowessError> {
    println!("{}", "=".repeat(80));
    println!("LOWESS Streaming Smoothing - Comprehensive Examples");
    println!("{}", "=".repeat(80));
    println!();

    // Run all example scenarios
    example_1_basic_chunked_processing()?;
    example_2_chunk_size_comparison()?;
    example_3_overlap_strategies()?;
    example_4_large_dataset_processing()?;
    example_5_outlier_handling()?;
    example_6_file_simulation()?;
    example_7_parallel_benchmark()?;
    example_8_sequential_benchmark()?;

    Ok(())
}

/// Example 1: Basic Chunked Processing
/// Demonstrates the fundamental streaming workflow
fn example_1_basic_chunked_processing() -> Result<(), LowessError> {
    println!("Example 1: Basic Chunked Processing");
    println!("{}", "-".repeat(80));

    // Generate test data: y = 2x + 1 with noise
    let n = 50;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.3).sin() * 2.0)
        .collect();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(2)
        .return_residuals()
        .adapter(Streaming)
        .chunk_size(15) // Process 15 points per chunk
        .overlap(5) // 5 points overlap between chunks
        .build()?;

    println!("Dataset: {} points", n);
    println!("Chunk size: 15, Overlap: 5");
    println!("Expected chunks: ~{}\n", (n as f64 / 10.0).ceil() as usize);

    let mut total_processed = 0;
    let chunk_size = 15;

    // Process in chunks
    for (chunk_idx, chunk_start) in (0..x.len()).step_by(chunk_size - 5).enumerate() {
        let chunk_end = (chunk_start + chunk_size).min(x.len());
        let x_chunk = &x[chunk_start..chunk_end];
        let y_chunk = &y[chunk_start..chunk_end];

        let result = processor.process_chunk(x_chunk, y_chunk)?;

        if !result.x.is_empty() {
            total_processed += result.x.len();
            println!(
                "Chunk {}: Processed {} points (x: {:.1} to {:.1})",
                chunk_idx,
                result.x.len(),
                result.x.first().unwrap(),
                result.x.last().unwrap()
            );
        }
    }

    // Finalize to get remaining points
    let final_result = processor.finalize()?;
    if !final_result.x.is_empty() {
        total_processed += final_result.x.len();
        println!(
            "Finalize: Processed {} remaining points (x: {:.1} to {:.1})",
            final_result.x.len(),
            final_result.x.first().unwrap(),
            final_result.x.last().unwrap()
        );
    }

    println!("\nTotal points processed: {}/{}", total_processed, n);

    /* Expected Output:
    Dataset: 50 points
    Chunk size: 15, Overlap: 5
    Expected chunks: ~5

    Chunk 0: Processed 10 points (x: 0.0 to 9.0)
    Chunk 1: Processed 10 points (x: 10.0 to 19.0)
    Chunk 2: Processed 10 points (x: 20.0 to 29.0)
    Chunk 3: Processed 10 points (x: 30.0 to 39.0)
    Finalize: Processed 10 remaining points (x: 40.0 to 49.0)

    Total points processed: 50/50
    */

    println!();
    Ok(())
}

/// Example 2: Chunk Size Comparison
/// Shows how different chunk sizes affect processing
fn example_2_chunk_size_comparison() -> Result<(), LowessError> {
    println!("Example 2: Chunk Size Comparison");
    println!("{}", "-".repeat(80));

    // Generate test data
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let chunk_configs = vec![
        (20, 5, "Small chunks"),
        (50, 10, "Medium chunks"),
        (80, 15, "Large chunks"),
    ];

    for (chunk_size, overlap, description) in chunk_configs {
        println!(
            "{} (size: {}, overlap: {})",
            description, chunk_size, overlap
        );

        let mut processor = Lowess::<f64>::new()
            .fraction(0.5)
            .iterations(1)
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .build()?;

        let mut chunk_count = 0;
        let mut total_processed = 0;

        for chunk_start in (0..x.len()).step_by(chunk_size - overlap) {
            let chunk_end = (chunk_start + chunk_size).min(x.len());
            let x_chunk = &x[chunk_start..chunk_end];
            let y_chunk = &y[chunk_start..chunk_end];

            let result = processor.process_chunk(x_chunk, y_chunk)?;

            if !result.x.is_empty() {
                chunk_count += 1;
                total_processed += result.x.len();
            }
        }

        let final_result = processor.finalize()?;
        if !final_result.x.is_empty() {
            chunk_count += 1;
            total_processed += final_result.x.len();
        }

        println!("  Chunks processed: {}", chunk_count);
        println!("  Total points: {}\n", total_processed);
    }

    /* Expected Output:
    Small chunks (size: 20, overlap: 5)
      Chunks processed: 7
      Total points: 100

    Medium chunks (size: 50, overlap: 10)
      Chunks processed: 3
      Total points: 100

    Large chunks (size: 80, overlap: 15)
      Chunks processed: 2
      Total points: 100
    */

    println!();
    Ok(())
}

/// Example 3: Overlap Strategies
/// Demonstrates different overlap configurations
fn example_3_overlap_strategies() -> Result<(), LowessError> {
    println!("Example 3: Overlap Strategies");
    println!("{}", "-".repeat(80));

    // Generate test data with discontinuity
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| {
            if xi < 30.0 {
                2.0 * xi
            } else {
                2.0 * xi + 10.0 // Jump at x=30
            }
        })
        .collect();

    let overlap_configs = vec![
        (0, "No overlap (fastest, potential edge artifacts)"),
        (5, "Small overlap (balanced)"),
        (10, "Large overlap (smoothest transitions)"),
    ];

    let chunk_size = 20;

    for (overlap, description) in overlap_configs {
        println!("{}", description);

        let mut processor = Lowess::<f64>::new()
            .fraction(0.5)
            .iterations(2)
            .adapter(Streaming)
            .chunk_size(chunk_size)
            .overlap(overlap)
            .build()?;

        let mut results = Vec::new();

        for chunk_start in (0..x.len()).step_by(chunk_size.saturating_sub(overlap)) {
            let chunk_end = (chunk_start + chunk_size).min(x.len());
            let x_chunk = &x[chunk_start..chunk_end];
            let y_chunk = &y[chunk_start..chunk_end];

            let result = processor.process_chunk(x_chunk, y_chunk)?;
            if !result.x.is_empty() {
                results.push(result);
            }
        }

        let final_result = processor.finalize()?;
        if !final_result.x.is_empty() {
            results.push(final_result);
        }

        println!("  Chunks produced: {}", results.len());
        println!(
            "  Total points: {}\n",
            results.iter().map(|r| r.x.len()).sum::<usize>()
        );
    }

    /* Expected Output:
    No overlap (fastest, potential edge artifacts)
      Chunks produced: 3
      Total points: 60

    Small overlap (balanced)
      Chunks produced: 4
      Total points: 60

    Large overlap (smoothest transitions)
      Chunks produced: 5
      Total points: 60
    */

    println!();
    Ok(())
}

/// Example 4: Large Dataset Processing
/// Simulates processing a very large dataset
fn example_4_large_dataset_processing() -> Result<(), LowessError> {
    println!("Example 4: Large Dataset Processing");
    println!("{}", "-".repeat(80));

    let n = 10000; // Simulate 10K points
    println!("Processing {} data points in streaming mode...", n);
    println!("(Simulating a dataset too large for memory)\n");

    let mut processor = Lowess::<f64>::new()
        .fraction(0.3)
        .iterations(2)
        .return_residuals()
        .adapter(Streaming)
        .chunk_size(500) // Process 500 points at a time
        .overlap(50) // 50 points overlap
        .build()?;

    let chunk_size = 500;
    let overlap = 50;
    let mut total_processed = 0;
    let mut chunk_count = 0;

    // Simulate streaming from a large data source
    for chunk_start in (0..n).step_by(chunk_size - overlap) {
        let chunk_end = (chunk_start + chunk_size).min(n);

        // Generate chunk on-the-fly (simulating reading from disk/network)
        let x_chunk: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();
        let y_chunk: Vec<f64> = x_chunk
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.01).sin() * 10.0)
            .collect();

        let result = processor.process_chunk(&x_chunk, &y_chunk)?;

        if !result.x.is_empty() {
            chunk_count += 1;
            total_processed += result.x.len();

            // Print progress every 5 chunks
            if chunk_count % 5 == 0 {
                println!(
                    "Progress: {} chunks processed, {} points smoothed (x: {:.0} to {:.0})",
                    chunk_count,
                    total_processed,
                    result.x.first().unwrap(),
                    result.x.last().unwrap()
                );
            }
        }
    }

    // Finalize
    let final_result = processor.finalize()?;
    if !final_result.x.is_empty() {
        chunk_count += 1;
        total_processed += final_result.x.len();
    }

    println!("\nProcessing complete!");
    println!("Total chunks: {}", chunk_count);
    println!("Total points processed: {}/{}", total_processed, n);
    println!("Memory efficiency: Constant (chunk size = 500 points)");

    /* Expected Output:
    Processing 10000 data points in streaming mode...
    (Simulating a dataset too large for memory)

    Progress: 5 chunks processed, 2250 points smoothed (x: 1800 to 2249)
    Progress: 10 chunks processed, 4500 points smoothed (x: 4050 to 4499)
    Progress: 15 chunks processed, 6750 points smoothed (x: 6300 to 6749)
    Progress: 20 chunks processed, 9000 points smoothed (x: 8550 to 8999)

    Processing complete!
    Total chunks: 23
    Total points processed: 10000/10000
    Memory efficiency: Constant (chunk size = 500 points)
    */

    println!();
    Ok(())
}

/// Example 5: Outlier Handling in Streaming Mode
/// Demonstrates robust smoothing with chunked data
fn example_5_outlier_handling() -> Result<(), LowessError> {
    println!("Example 5: Outlier Handling in Streaming Mode");
    println!("{}", "-".repeat(80));

    // Generate data with outliers
    let n = 100;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let base = 2.0 * xi + 1.0;
            // Add outliers at specific positions
            if i == 25 || i == 50 || i == 75 {
                base + 50.0 // Outlier
            } else {
                base + (xi * 0.2).sin() * 2.0
            }
        })
        .collect();

    println!("Dataset: {} points with 3 outliers", n);
    println!("Testing robustness methods:\n");

    let methods = vec![(Bisquare, "Bisquare"), (Huber, "Huber"), (Talwar, "Talwar")];

    for (method, name) in methods {
        println!("Using {} robustness:", name);

        let mut processor = Lowess::<f64>::new()
            .fraction(0.5)
            .iterations(5) // More iterations for better outlier handling
            .robustness_method(method)
            .return_residuals()
            .adapter(Streaming)
            .chunk_size(30)
            .overlap(10)
            .build()?;

        let mut large_residuals = 0;
        let chunk_size = 30;

        for chunk_start in (0..x.len()).step_by(chunk_size - 10) {
            let chunk_end = (chunk_start + chunk_size).min(x.len());
            let x_chunk = &x[chunk_start..chunk_end];
            let y_chunk = &y[chunk_start..chunk_end];

            let result = processor.process_chunk(x_chunk, y_chunk)?;

            if !result.x.is_empty() {
                if let Some(residuals) = &result.residuals {
                    large_residuals += residuals.iter().filter(|&&r| r.abs() > 10.0).count();
                }
            }
        }

        let final_result = processor.finalize()?;
        if let Some(residuals) = &final_result.residuals {
            large_residuals += residuals.iter().filter(|&&r| r.abs() > 10.0).count();
        }

        println!(
            "  Points with large residuals (|r| > 10): {}",
            large_residuals
        );
    }

    /* Expected Output:
    Dataset: 100 points with 3 outliers
    Testing robustness methods:

    Using Bisquare robustness:
      Points with large residuals (|r| > 10): 3
    Using Huber robustness:
      Points with large residuals (|r| > 10): 3
    Using Talwar robustness:
      Points with large residuals (|r| > 10): 3
    */

    println!();
    Ok(())
}

/// Example 6: File-Based Streaming Simulation
/// Simulates reading from a file and writing results incrementally
fn example_6_file_simulation() -> Result<(), LowessError> {
    println!("Example 6: File-Based Streaming Simulation");
    println!("{}", "-".repeat(80));
    println!("Simulating: Read from input.csv -> Smooth -> Write to output.csv\n");

    // Simulate file data
    let total_lines = 200;
    println!("Input file: {} data points", total_lines);

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(2)
        .return_residuals()
        .adapter(Streaming)
        .chunk_size(50)
        .overlap(10)
        .build()?;

    let chunk_size = 50;
    let mut output_lines = 0;

    println!("Processing in chunks...\n");

    // Simulate reading and processing file chunks
    for chunk_idx in 0..(total_lines / (chunk_size - 10)) {
        let chunk_start = chunk_idx * (chunk_size - 10);
        let chunk_end = (chunk_start + chunk_size).min(total_lines);

        // Simulate reading chunk from file
        let x_chunk: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();
        let y_chunk: Vec<f64> = x_chunk
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + (xi * 0.1).sin() * 3.0)
            .collect();

        println!(
            "Reading chunk {} (lines {} to {})",
            chunk_idx,
            chunk_start,
            chunk_end - 1
        );

        let result = processor.process_chunk(&x_chunk, &y_chunk)?;

        if !result.x.is_empty() {
            // Simulate writing to output file
            output_lines += result.x.len();
            println!(
                "  -> Writing {} smoothed points to output (total: {})",
                result.x.len(),
                output_lines
            );
        }
    }

    // Finalize and write remaining
    let final_result = processor.finalize()?;
    if !final_result.x.is_empty() {
        output_lines += final_result.x.len();
        println!(
            "\nFinalizing: Writing {} remaining points",
            final_result.x.len()
        );
    }

    println!("\nProcessing complete!");
    println!("Input lines: {}", total_lines);
    println!("Output lines: {}", output_lines);
    println!("Status: ✓ All data processed successfully");

    /* Expected Output:
    Simulating: Read from input.csv -> Smooth -> Write to output.csv

    Input file: 200 data points
    Processing in chunks...

    Reading chunk 0 (lines 0 to 49)
      -> Writing 40 smoothed points to output (total: 40)
    Reading chunk 1 (lines 40 to 89)
      -> Writing 40 smoothed points to output (total: 80)
    Reading chunk 2 (lines 80 to 129)
      -> Writing 40 smoothed points to output (total: 120)
    Reading chunk 3 (lines 120 to 169)
      -> Writing 40 smoothed points to output (total: 160)
    Reading chunk 4 (lines 160 to 199)

    Finalizing: Writing 40 remaining points

    Processing complete!
    Input lines: 200
    Output lines: 200
    Status: ✓ All data processed successfully
    */

    println!();
    Ok(())
}

/// Example 7: Parallel Streaming Benchmark
/// Measure execution time for a large dataset using the parallel Streaming adapter
fn example_7_parallel_benchmark() -> Result<(), LowessError> {
    println!("Example 7: Benchmark (Parallel Streaming)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    println!("Processing {} data points in parallel streaming mode...", n);

    let start = std::time::Instant::now();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Streaming)
        .chunk_size(1000) // Process 1000 points per chunk
        .overlap(100) // 100 points overlap
        .parallel(true) // Enable parallel execution
        .build()?;

    let chunk_size = 1000;
    let overlap = 100;
    let mut total_processed = 0;

    // Process in chunks
    for chunk_start in (0..n).step_by(chunk_size - overlap) {
        let chunk_end = (chunk_start + chunk_size).min(n);

        // Generate chunk on-the-fly
        let x_chunk: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();
        let y_chunk: Vec<f64> = x_chunk
            .iter()
            .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
            .collect();

        let result = processor.process_chunk(&x_chunk, &y_chunk)?;
        total_processed += result.x.len();
    }

    // Finalize
    let final_result = processor.finalize()?;
    total_processed += final_result.x.len();

    let duration = start.elapsed();

    println!("Processed {} points in {:?}", total_processed, duration);
    println!("Execution mode: Parallel Streaming");
    println!("Chunk size: {}, Overlap: {}", chunk_size, overlap);

    println!();
    Ok(())
}

/// Example 8: Sequential Streaming Benchmark
/// Measure execution time for a large dataset using the sequential Streaming adapter
fn example_8_sequential_benchmark() -> Result<(), LowessError> {
    println!("Example 8: Benchmark (Sequential Streaming)");
    println!("{}", "-".repeat(80));

    // Generate a larger synthetic dataset
    let n = 10_000;
    println!(
        "Processing {} data points in sequential streaming mode...",
        n
    );

    let start = std::time::Instant::now();

    let mut processor = Lowess::<f64>::new()
        .fraction(0.5)
        .iterations(3)
        .adapter(Streaming)
        .chunk_size(1000) // Process 1000 points per chunk
        .overlap(100) // 100 points overlap
        .parallel(false) // Disable parallel execution
        .build()?;

    let chunk_size = 1000;
    let overlap = 100;
    let mut total_processed = 0;

    // Process in chunks
    for chunk_start in (0..n).step_by(chunk_size - overlap) {
        let chunk_end = (chunk_start + chunk_size).min(n);

        // Generate chunk on-the-fly
        let x_chunk: Vec<f64> = (chunk_start..chunk_end).map(|i| i as f64).collect();
        let y_chunk: Vec<f64> = x_chunk
            .iter()
            .map(|&xi| (xi * 0.1).sin() + (xi * 0.01).cos())
            .collect();

        let result = processor.process_chunk(&x_chunk, &y_chunk)?;
        total_processed += result.x.len();
    }

    // Finalize
    let final_result = processor.finalize()?;
    total_processed += final_result.x.len();

    let duration = start.elapsed();

    println!("Processed {} points in {:?}", total_processed, duration);
    println!("Execution mode: Sequential Streaming");
    println!("Chunk size: {}, Overlap: {}", chunk_size, overlap);

    println!();
    Ok(())
}
