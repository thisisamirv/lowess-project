# Python API

API reference for the `fastlowess` Python package.

## Installation

```bash
pip install fastlowess
```

---

## Functions

### smooth

Main function for batch smoothing.

```python
fastlowess.smooth(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.67,
    iterations: int = 3,
    delta: float | None = None,
    parallel: bool = True,
    weight_function: str = "tricube",
    robustness_method: str = "bisquare",
    scaling_method: str = "mad",
    zero_weight_fallback: str = "use_local_mean",
    boundary_policy: str = "extend",
    auto_converge: float | None = None,
    return_residuals: bool = False,
    return_diagnostics: bool = False,
    return_robustness_weights: bool = False,
    confidence_intervals: float | None = None,
    prediction_intervals: float | None = None,
    cv_method: str | None = None,
    cv_k: int = 5,
    cv_fractions: list[float] | None = None,
    cv_seed: int | None = None,
) -> dict
```

**Parameters:**

| Parameter    | Type  | Default  | Description             |
|--------------|-------|----------|-------------------------|
| `x`          | array | required | Independent variable    |
| `y`          | array | required | Dependent variable      |
| `fraction`   | float | 0.67     | Smoothing span (0, 1]   |
| `iterations` | int   | 3        | Robustness iterations   |
| `delta`      | float | auto     | Interpolation threshold |
| `parallel`   | bool  | True     | Enable parallelism      |

**Returns:** `dict` with keys:

| Key                  | Type    | Description                         |
|----------------------|---------|-------------------------------------|
| `x`                  | ndarray | Input x values                      |
| `y`                  | ndarray | Smoothed y values                   |
| `fraction_used`      | float   | Actual fraction used                |
| `residuals`          | ndarray | If `return_residuals=True`          |
| `confidence_lower`   | ndarray | If `confidence_intervals` set       |
| `confidence_upper`   | ndarray | If `confidence_intervals` set       |
| `prediction_lower`   | ndarray | If `prediction_intervals` set       |
| `prediction_upper`   | ndarray | If `prediction_intervals` set       |
| `robustness_weights` | ndarray | If `return_robustness_weights=True` |
| `diagnostics`        | dict    | If `return_diagnostics=True`        |
| `cv_scores`          | list    | If cross-validation used            |

**Example:**

```python
import fastlowess as fl
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

result = fl.smooth(x, y, fraction=0.3, iterations=3)
print(result["y"])
```

---

### smooth_streaming

Streaming mode for large datasets.

```python
fastlowess.smooth_streaming(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.67,
    iterations: int = 3,
    chunk_size: int = 5000,
    overlap: int = 500,
    merge_strategy: str = "average",
    parallel: bool = True,
    **kwargs  # Same as smooth()
) -> dict
```

**Additional Parameters:**

| Parameter        | Type | Default   | Description            |
|------------------|------|-----------|------------------------|
| `chunk_size`     | int  | 5000      | Points per chunk       |
| `overlap`        | int  | 500       | Overlap between chunks |
| `merge_strategy` | str  | "average" | How to merge overlaps  |

**Example:**

```python
# Process 1 million points
x = np.linspace(0, 1000, 1_000_000)
y = np.sin(x / 100) + np.random.normal(0, 0.1, 1_000_000)

result = fl.smooth_streaming(x, y, chunk_size=10000, overlap=1000)
```

---

### smooth_online

Online mode for real-time data.

```python
fastlowess.smooth_online(
    x: ArrayLike,
    y: ArrayLike,
    fraction: float = 0.2,
    window_capacity: int = 100,
    min_points: int = 2,
    iterations: int = 3,
    update_mode: str = "incremental",
    **kwargs  # Same as smooth()
) -> dict
```

**Additional Parameters:**

| Parameter         | Type | Default       | Description          |
|-------------------|------|---------------|----------------------|
| `window_capacity` | int  | 100           | Max points in window |
| `min_points`      | int  | 2             | Points before output |
| `update_mode`     | str  | "incremental" | Update strategy      |

**Example:**

```python
# Sensor data simulation
sensor_times = np.arange(100)
sensor_values = 20 + 5 * np.sin(sensor_times / 10) + np.random.normal(0, 1, 100)

result = fl.smooth_online(sensor_times, sensor_values, window_capacity=25)
```

---

## String Options

### weight_function

- `"tricube"` (default)
- `"epanechnikov"`
- `"gaussian"`
- `"biweight"`
- `"cosine"`
- `"triangle"`
- `"uniform"`

### robustness_method

- `"bisquare"` (default)
- `"huber"`
- `"talwar"`

### boundary_policy

- `"extend"` (default)
- `"reflect"`
- `"zero"`
- `"no_boundary"`

### merge_strategy

- `"average"` (default)
- `"left"`
- `"right"`
- `"weighted"`

### update_mode

- `"incremental"` (default)
- `"full"`

---

## Diagnostics

When `return_diagnostics=True`, the result includes:

```python
result["diagnostics"] = {
    "rmse": float,        # Root Mean Square Error
    "mae": float,         # Mean Absolute Error
    "r_squared": float,   # R² coefficient
    "residual_sd": float, # Residual standard deviation
    "effective_df": float # Effective degrees of freedom
}
```

---

## Exceptions

```python
class LowessError(Exception):
    """Base exception for LOWESS errors."""
    pass
```

Common error conditions:

- Mismatched array lengths
- Invalid fraction (not in (0, 1])
- Insufficient data points
- Empty input arrays
