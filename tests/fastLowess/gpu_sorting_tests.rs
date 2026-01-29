#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use fastLowess::prelude::*;

    #[test]
    fn test_gpu_unsorted_input() {
        // Create unsorted data
        // y = 2x, but shuffled order
        let x = vec![2.0, 0.0, 4.0, 1.0, 3.0];
        let y = vec![4.0, 0.0, 8.0, 2.0, 6.0];

        // Fit using GPU
        // Note: fit returns Result<LowessResult, Error>
        let result = Lowess::new()
            .backend(GPU)
            .fraction(0.5)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();

        // The GPU sorts input by X: [0.0, 1.0, 2.0, 3.0, 4.0]
        // So the output Y should be [0.0, 2.0, 4.0, 6.0, 8.0]
        // Note: The OUTPUT order corresponds to the SORTED X, not the original unsorted X.

        println!("GPU Result X (implied): [0.0, 1.0, 2.0, 3.0, 4.0]");
        println!("GPU Result Y: {:?}", result.y);

        let expected_y_sorted = vec![0.0, 2.0, 4.0, 6.0, 8.0];

        for (i, &val) in result.y.iter().enumerate() {
            assert_abs_diff_eq!(val, expected_y_sorted[i], epsilon = 1e-4);
        }
    }
}
