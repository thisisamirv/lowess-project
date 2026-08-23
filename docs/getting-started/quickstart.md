<!-- markdownlint-disable MD024 MD046 -->
# Quick Start

Get up and running with LOWESS in minutes.

## Basic Smoothing

Smooth a noisy sine wave — the kind of signal where LOWESS shines. Each example recovers the underlying trend from 100 points of Gaussian noise.

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np

    # 100-point noisy sine wave
    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    model = fl.Lowess(fraction=0.3, iterations=3)
    result = model.fit(x, y)

    print(f"First smoothed value: {result.y[0]:.4f}  (true: {np.sin(x[0]):.4f})")
    ```

=== "Rust"

    ```rust
    use fastLowess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LowessError> {
        // 100-point noisy sine wave (deterministic)
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().enumerate()
            .map(|(i, &xi)| xi.sin() + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.3)
            .collect();

        let model = Lowess::new()
            .fraction(0.3)
            .iterations(3)
            .build()?;

        let result = model.fit(&x, &y)?;
        println!("First smoothed: {:.4}  (true: {:.4})", result.y[0], x[0].sin());
        Ok(())
    }
    ```
=== "Node.js"

    ```javascript
    const { Lowess } = require('fastlowess');

    // 100-point noisy sine wave
    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({ fraction: 0.3, iterations: 3 });
    const result = model.fit(x, y);

    console.log(`First smoothed: ${result.y[0].toFixed(4)}  (true: ${Math.sin(x[0]).toFixed(4)})`);
    ```

=== "WebAssembly"

    ```javascript
    const { Lowess } = require('./fastlowess_wasm.js');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({ fraction: 0.3, iterations: 3 });
    const result = model.fit(x, y);

    console.log(`First smoothed: ${result.y[0].toFixed(4)}`);
    ```

=== "C++"

    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        // 100-point noisy sine wave (deterministic)
        const int n = 100;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 2 * M_PI / (n - 1);
            y[i] = std::sin(x[i]) + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.6;
        }

        fastlowess::Lowess model({ .fraction = 0.3, .iterations = 3 });
        auto result = model.fit(x, y).value();

        std::cout << "First smoothed: " << result.y_vector()[0]
                  << "  (true: " << std::sin(x[0]) << ")\n";
        return 0;
    }
    ```

---

## With Confidence Intervals

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + rng.normal(0, 0.3, 100)

    model = fl.Lowess(
        fraction=0.5,
        iterations=3,
        confidence_intervals=0.95,
        prediction_intervals=0.95,
        return_diagnostics=True
    )
    result = model.fit(x, y)

    print("Smoothed:", result.y)
    print("CI Lower:", result.confidence_lower)
    print("CI Upper:", result.confidence_upper)
    print("R²:", result.diagnostics.r_squared)
    ```

=== "Rust"

    ```rust
    use fastLowess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LowessError> {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();


        let model = Lowess::new()
            .fraction(0.5)
            .iterations(3)
            .confidence_intervals(0.95)  // 95% CI
            .prediction_intervals(0.95)  // 95% PI
            .return_diagnostics()
            .build()?;

        let result = model.fit(&x, &y)?;

        // Access intervals
        if let Some(ci_lower) = &result.confidence_lower {
            println!("CI Lower: {:?}", ci_lower);
        }

        Ok(())
    }
    ```
=== "Node.js"

    ```javascript
    const { Lowess } = require('fastlowess');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i*7+3)%17)/17-0.5)*0.6);

    const model = new Lowess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
        return_diagnostics: true
    });
    const result = model.fit(x, y);

    console.log("Smoothed:", result.y);
    console.log("CI Lower:", result.confidence_lower);
    console.log("CI Upper:", result.confidence_upper);
    console.log("R²:", result.diagnostics.r_squared);
    ```

=== "WebAssembly"

    ```javascript
    const { Lowess } = require('./fastlowess_wasm.js');

    const n = 100;
    const x = Float64Array.from({ length: n }, (_, i) => i * 2 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) => Math.sin(xi) + (((i * 7 + 3) % 17) / 17 - 0.5) * 0.6);

    const model = new Lowess({
        fraction: 0.5,
        iterations: 3,
        confidence_intervals: 0.95,
        prediction_intervals: 0.95,
        return_diagnostics: true
    });
    const result = model.fit(x, y);

    console.log("Smoothed:", result.y);
    console.log("CI Lower:", result.confidence_lower);
    console.log("CI Upper:", result.confidence_upper);
    console.log("R²:", result.diagnostics.r_squared);
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

        fastlowess::LowessOptions options;
        options.fraction = 0.5;
        options.iterations = 3;
        options.confidence_intervals = 0.95;
        options.prediction_intervals = 0.95;
        options.return_diagnostics = true;

        fastlowess::Lowess model(options);
        auto result = model.fit(x, y).value();

        // Access standard C++ vectors
        auto lower = result.confidence_lower();
        auto upper = result.confidence_upper();
        double r2 = result.diagnostics().r_squared();

        return 0;
    }
    ```

---

## Handling Outliers

LOWESS can robustly handle outliers through iterative reweighting:

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    trend_true = 10 + 0.5 * t + 3 * np.sin(t / 10)
    y = trend_true + np.random.normal(0, 3, len(t))

    x_out = np.linspace(1, 6, 6)
    y_with_outlier = np.array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0])

    model = fl.Lowess(
        fraction=0.5,
        iterations=5,
        robustness_method="bisquare",
        return_robustness_weights=True
    )
    result = model.fit(x_out, y_with_outlier)

    # Check which points were downweighted
    for i, w in enumerate(result.robustness_weights):
        if w < 0.5:
            print(f"Point {i} is likely an outlier (weight: {w:.3f})")
    ```

=== "Rust"

    ```rust
    use fastLowess::prelude::*;
    use std::f64::consts::TAU;

    fn main() -> Result<(), LowessError> {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * TAU / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1).collect();

        // Data with an outlier at position 3
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y_with_outlier = vec![2.0, 4.0, 6.0, 50.0, 10.0, 12.0];  // 50.0 is outlier

        let model = Lowess::new()
            .fraction(0.5)
            .iterations(5)                    // More iterations for outliers
            .robustness_method("bisquare")    // Default, smooth downweighting
            .return_robustness_weights()      // See which points were downweighted
            .build()?;

        let result = model.fit(&x, &y_with_outlier)?;

        // Outliers will have low robustness weights
        if let Some(weights) = &result.robustness_weights {
            for (i, w) in weights.iter().enumerate() {
                if *w < 0.5 {
                    println!("Point {} is likely an outlier (weight: {:.3})", i, w);
                }
            }
        }

        Ok(())
    }
    ```
=== "Node.js"

    ```javascript
    const { Lowess } = require('fastlowess');

    const xOut = new Float64Array([1, 2, 3, 4, 5, 6]);
    const yWithOutlier = new Float64Array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0]);

    const model = new Lowess({
        fraction: 0.5,
        iterations: 5,
        robustness_method: "bisquare",
        return_robustness_weights: true
    });
    const result = model.fit(xOut, yWithOutlier);

    // Outliers will have low robustness weights
    result.robustness_weights.forEach((w, i) => {
        if (w < 0.5) {
            console.log(`Point ${i} is likely an outlier (weight: ${w.toFixed(3)})`);
        }
    });
    ```

=== "WebAssembly"

    ```javascript
    const { Lowess } = require('./fastlowess_wasm.js');

    // Data with an outlier at position 3
    const x = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const yWithOutlier = new Float64Array([2.0, 4.0, 6.0, 50.0, 10.0, 12.0]);

    const model = new Lowess({
        fraction: 0.5,
        iterations: 5,
        robustness_method: "bisquare",
        return_robustness_weights: true
    });
    const result = model.fit(x, yWithOutlier);

    // Outliers will have low robustness weights
    result.robustness_weights.forEach((w, i) => {
        if (w < 0.5) {
            console.log(`Point ${i} is likely an outlier (weight: ${w.toFixed(3)})`);
        }
    });
    ```

=== "C++"

    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        // Data with an outlier at index 3
        std::vector<double> x_out = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        std::vector<double> y_outlier = {2.0, 4.0, 6.0, 50.0, 10.0, 12.0};

        fastlowess::LowessOptions options;
        options.fraction = 0.5;
        options.iterations = 5;
        options.robustness_method = "bisquare";
        options.return_robustness_weights = true;

        fastlowess::Lowess model(options);
        auto result = model.fit(x_out, y_outlier).value();

        // Check weights
        auto weights = result.robustness_weights();
        for (size_t i = 0; i < weights.size(); ++i) {
            if (weights[i] < 0.5) {
                std::cout << "Point " << i << " is outlier (weight: " << weights[i] << ")\n";
            }
        }

        return 0;
    }
    ```

---

## Streaming Mode

For datasets too large to fit in memory, stream them in fixed-size chunks with overlap.

=== "Python"

    ```python
    import fastlowess as fl
    import numpy as np

    rng = np.random.default_rng(42)
    x = np.linspace(0, 10 * np.pi, 5000)
    y = np.sin(x / np.pi) * np.exp(-x / 30) + rng.normal(0, 0.15, 5000)

    model = fl.StreamingLowess(
        fraction=0.2,
        chunk_size=1000,
        overlap=100,
        merge_strategy="weighted_average",
    )

    chunk_size = 1000
    for start in range(0, 4001, chunk_size):
        end = min(start + chunk_size, len(x))
        model.process_chunk(x[start:end], y[start:end])
    result = model.finalize()
    print(f"Smoothed {len(result.y)} points in streaming mode")
    ```

=== "Rust"

    ```rust
    use fastLowess::prelude::*;
    use std::f64::consts::PI;

    fn main() -> Result<(), LowessError> {
        let n = 5_000usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 10.0 * PI / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().enumerate()
            .map(|(i, &xi)| (xi / PI).sin() * (-xi / 30.0).exp()
                           + ((i * 7 + 3) as f64 % 1.7 - 0.85) * 0.15)
            .collect();

        let mut model = StreamingLowess::new()
            .fraction(0.2)
            .chunk_size(1000)
            .overlap(100)
            .build()?;

        for chunk in x.chunks(1000).zip(y.chunks(1000)) {
            model.process_chunk(chunk.0, chunk.1)?;
        }
        let result = model.finalize()?;
        println!("Smoothed {} points", result.y.len());
        Ok(())
    }
    ```
=== "Node.js"

    ```javascript
    const { StreamingLowess } = require('fastlowess');

    const n = 5000;
    const x = Float64Array.from({ length: n }, (_, i) => i * 10 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) =>
        Math.sin(xi / Math.PI) * Math.exp(-xi / 30) +
        (((i * 7 + 3) % 17) / 17 - 0.5) * 0.3
    );

    const model = new StreamingLowess(
        { fraction: 0.2 },
        { chunk_size: 1000, overlap: 100, merge_strategy: 'weighted_average' }
    );

    const chunk_size = 1000;
    for (let start = 0; start <= 4000; start += chunk_size) {
        const end = Math.min(start + chunk_size, n);
        model.process_chunk(x.slice(start, end), y.slice(start, end));
    }
    const result = model.finalize();
    console.log(`Smoothed ${result.y.length} points in streaming mode`);
    ```

=== "WebAssembly"

    ```javascript
    const { StreamingLowess } = require('./fastlowess_wasm.js');

    const n = 5000;
    const x = Float64Array.from({ length: n }, (_, i) => i * 10 * Math.PI / (n - 1));
    const y = Float64Array.from(x, (xi, i) =>
        Math.sin(xi / Math.PI) * Math.exp(-xi / 30) +
        (((i * 7 + 3) % 17) / 17 - 0.5) * 0.3
    );

    const model = new StreamingLowess(
        { fraction: 0.2 },
        { chunk_size: 1000, overlap: 100, merge_strategy: 'weighted_average' }
    );

    const chunk_size = 1000;
    for (let start = 0; start <= 4000; start += chunk_size) {
        const end = Math.min(start + chunk_size, n);
        model.process_chunk(x.slice(start, end), y.slice(start, end));
    }
    const result = model.finalize();
    console.log(`Smoothed ${result.y.length} points`);
    ```

=== "C++"

    ```cpp
    #include <fastlowess.hpp>
    #include <cmath>
    #include <iostream>
    #include <vector>

    int main() {
        const int n = 5000;
        std::vector<double> x(n), y(n);
        for (int i = 0; i < n; ++i) {
            x[i] = i * 10 * M_PI / (n - 1);
            y[i] = std::sin(x[i] / M_PI) * std::exp(-x[i] / 30.0)
                 + ((i * 7 + 3) % 17 / 17.0 - 0.5) * 0.3;
        }

        fastlowess::StreamingOptions opts;
        opts.fraction   = 0.2;
        opts.chunk_size = 1000;
        opts.overlap    = 100;

        fastlowess::StreamingLowess model(opts);

        for (int start = 0; start <= 4000; start += 1000) {
            int end = std::min(start + 1000, n);
            model.process_chunk(
                std::vector<double>(x.begin() + start, x.begin() + end),
                std::vector<double>(y.begin() + start, y.begin() + end)
            );
        }
        auto result = model.finalize().value();
        std::cout << "Smoothed " << result.y_vector().size() << " points\n";
        return 0;
    }
    ```

---

## Next Steps

| Topic | Link |
| --- | --- |
| How LOWESS works | [Concepts](concepts.md) |
| All parameters explained | [Parameters](../user-guide/parameters.md) |
| Batch vs Streaming vs Online | [Execution Modes](../user-guide/adapters.md) |
| Edge handling | [Boundary](../user-guide/boundary.md) |
| Outlier handling in depth | [Robustness](../user-guide/robustness.md) |
| Full API per language | [API Reference](../api/index.md) |
