Execution Modes
===============

`fastlowess` offers three execution modes designed for different data scales and use cases. Each mode corresponds to an underlying "Adapter" in the Rust engine.

Execution Mode Comparison
-------------------------

+-----------+----------------------------------------------+--------------+----------------------------------+-----------------------------+
| Mode      | Use Case                                     | Parallel     | Features                         | Limitations                 |
+===========+==============================================+==============+==================================+=============================+
| Batch     | Dataset fits in memory (<1M points)          | ✅ Yes       | ✅ Full (CV, Intervals, Diag)    | ❌ High memory usage        |
+-----------+----------------------------------------------+--------------+----------------------------------+-----------------------------+
| Streaming | Very large datasets (>1M points)             | ✅ Yes       | ✅ Low memory, Diag, Weights     | ❌ No intervals, CV         |
+-----------+----------------------------------------------+--------------+----------------------------------+-----------------------------+
| Online    | Real-time streams, Sensor data               | ❌ No*       | ✅ Sliding window, Weights       | ❌ Local only               |
+-----------+----------------------------------------------+--------------+----------------------------------+-----------------------------+

*\*Online mode defaults to sequential execution for minimal latency, but can be parallelized if needed.*

1. Batch Mode (Standard)
------------------------

This is the default mode used by `fastlowess.smooth()`. It processes the entire dataset in memory.

*   **Best for:** Standard analysis, exploratory data analysis, plotting.
*   **Performance:** Uses Rayon for multi-threaded execution.

2. Streaming Mode (Large Datasets)
----------------------------------

For datasets that are too large to fit in memory or require efficient batch processing, use `fastlowess.smooth_streaming()`. This method processes data in chunks, maintaining constant memory usage.

.. code-block:: python

    import fastlowess
    import numpy as np

    # 1. Create a large synthetic dataset (50k points)
    n = 50_000
    x = np.arange(n, dtype=float)
    y = 2.0 * x + 1.0 + np.sin(x * 0.01) * 10.0

    # 2. Process in chunks
    # This uses constant memory regardless of dataset size
    result = fastlowess.smooth_streaming(
        x, y,
        fraction=0.3,
        chunk_size=5_000,   # Process 5k points at a time
        overlap=500,        # 10% overlap ensures smoothness at boundaries
        parallel=True       # Use multiple cores
    )

    print(f"Processed {len(result.y)} points")

    # Output:
    # Processed 50000 points


3. Online Mode (Real-time)
--------------------------

For real-time data streams, such as sensor data, use `fastlowess.smooth_online()`. This maintains a sliding window of the most recent data points.

.. code-block:: python

    import fastlowess
    import numpy as np
    import math

    # 1. Simulate temperature sensor data (24 hours)
    n = 24
    x = np.arange(n, dtype=float)
    base_temp = 20.0
    daily_cycle = 5.0 * np.sin(x * math.pi / 12.0)
    noise = np.random.normal(0, 0.5, n)
    y = base_temp + daily_cycle + noise

    # 2. Process with sliding window (Real-time simulation)
    result = fastlowess.smooth_online(
        x, y,
        fraction=0.4,
        window_capacity=12,
        min_points=2,
        update_mode="incremental", # "incremental" (default) or "full"
        parallel=False
    )

    print(f"Smoothed {len(result.y)} points")

    # Output:
    # Smoothed 24 points

Performance & Tuning
--------------------

### Parallel vs Sequential

Batch and Streaming modes are parallel by default. `fastlowess` scales linearly with the number of cores for large datasets.

**Typical Speedup Factors (vs Single-threaded):**

+--------------+-----------+----------------+
| Dataset Size | CPU Cores | Speedup Factor |
+==============+===========+================+
| 1,000        | 4         | ~1.5 - 2x      |
+--------------+-----------+----------------+
| 10,000       | 4         | ~2.5 - 3x      |
+--------------+-----------+----------------+
| 100,000      | 8         | ~5 - 6x        |
+--------------+-----------+----------------+

### Delta Optimization

For large datasets in Batch mode, you can enable `delta` optimization. This skips re-calculation for points that are very close to each other, using interpolation instead.

.. code-block:: python

    # Delta optimization enabled
    result = fastlowess.smooth(
        x, y,
        fraction=0.5,
        delta=0.01 * (x.max() - x.min()) # Interpolate within 1% range
    )

**Recommendation:** Use `delta` for N > 10,000 if slight approximation (interpolation error) is acceptable.
