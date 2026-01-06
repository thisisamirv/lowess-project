Examples
========

This page provides practical examples of using ``fastlowess`` in various execution modes, demonstrating key features like robust smoothing, interval calculation, streaming, and online updates.

You can find the full source code for these examples in the `examples/ <https://github.com/thisisamirv/fastLowess-py/tree/main/examples>`_ directory of the repository.

1. Batch Smoothing (Standard)
-----------------------------

**File:** `examples/batch_smoothing.py <https://github.com/thisisamirv/fastLowess-py/blob/main/examples/batch_smoothing.py>`_

Demonstrates the primary `smooth()` function, including:
- Basic vs. Robust smoothing
- Confidence and Prediction Intervals
- Cross-Validation
- Diagnostics

.. code-block:: python

    import numpy as np
    import fastlowess
    from fastlowess import smooth

    # 1. Basic Smoothing (Non-robust)
    # Fast, standard weighted least squares
    res_basic = smooth(x, y, iterations=0, fraction=0.05)

    # 2. Robust Smoothing (IRLS)
    # Resistant to outliers (using 'bisquare' weights by default)
    res_robust = smooth(
        x, y, 
        fraction=0.05, 
        iterations=3, 
        robustness_method="bisquare",
        return_robustness_weights=True
    )

    # 3. Uncertainty Quantification
    # Compute 95% Confidence and Prediction Intervals
    res_intervals = smooth(
        x, y, 
        fraction=0.05, 
        confidence_intervals=0.95, 
        prediction_intervals=0.95,
        return_diagnostics=True
    )
    
    # Access diagnostics
    print(f"R²: {res_intervals.diagnostics.r_squared:.4f}")

    # 4. Cross-Validation
    # Automatically find the best fraction
    res_cv = smooth(
        x, y, 
        cv_fractions=[0.05, 0.1, 0.2, 0.4], 
        cv_method="kfold"
    )
    print(f"Optimal fraction: {res_cv.fraction_used}")

2. Streaming (Large Datasets)
-----------------------------

**File:** `examples/streaming_smoothing.py <https://github.com/thisisamirv/fastLowess-py/blob/main/examples/streaming_smoothing.py>`_

Demonstrates `smooth_streaming()` for processing datasets that may not fit in memory or require chunked processing.

.. code-block:: python

    from fastlowess import smooth_streaming

    # Process a large dataset in chunks
    # This maintains constant memory usage regardless of total size
    res_stream = smooth_streaming(
        x, y, 
        fraction=0.01, 
        chunk_size=2000, 
        overlap=200,    # Overlap ensures smooth transitions between chunks
        parallel=True   # Use multi-core processing
    )

3. Online Smoothing (Real-time)
-------------------------------

**File:** `examples/online_smoothing.py <https://github.com/thisisamirv/fastLowess-py/blob/main/examples/online_smoothing.py>`_

Demonstrates `smooth_online()` for processing real-time data streams using a sliding window. Ideal for sensor data.

.. code-block:: python

    from fastlowess import smooth_online

    # 1. Full Update Mode
    # Recalculates the fit for the entire window at each step (O(N) per point)
    # Higher accuracy, simpler logic
    res_full = smooth_online(
        x, y, 
        fraction=0.3, 
        window_capacity=50, 
        update_mode="full"
    )

    # 2. Incremental Update Mode
    # Updates the existing fit (O(1) amortized per point)
    # Significantly faster for large windows
    res_inc = smooth_online(
        x, y, 
        fraction=0.3, 
        window_capacity=50, 
        update_mode="incremental"
    )

Running the Examples
--------------------

The examples are included in the repository and can be run directly:

.. code-block:: bash

    # Run batch smoothing example
    python examples/batch_smoothing.py

    # Run streaming example
    python examples/streaming_smoothing.py

    # Run online example
    python examples/online_smoothing.py

These scripts will generate plots in the ``examples/plots/`` directory.
