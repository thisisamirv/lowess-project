# Python API

API reference for the `fastlowess` Python package.

---

## Classes

### Lowess

Stateful class for batch smoothing.

```python
class fastlowess.Lowess(
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
)
```

**Methods:**

#### `fit`

Fits the model to the data and returns the result.

```python
def fit(x: ArrayLike, y: ArrayLike) -> LowessResult
```

**Parameters:**

| Parameter    | Type  | Default  | Description             |
|--------------|-------|----------|-------------------------|
| `fraction`   | float | 0.67     | Smoothing span (0, 1]   |
| `iterations` | int   | 3        | Robustness iterations   |
| `delta`      | float | auto     | Interpolation threshold |
| `parallel`   | bool  | True     | Enable parallelism      |

**Returns:** `LowessResult` object.

**Example:**

```python
import fastlowess as fl
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

lowess = fl.Lowess(fraction=0.3, iterations=3)
result = lowess.fit(x, y)
print(result.y)
```

---

### StreamingLowess

Stateful streaming mode for large datasets.

```python
class fastlowess.StreamingLowess(
    fraction: float = 0.67,
    iterations: int = 3,
    chunk_size: int = 5000,
    overlap: int = 500,
    merge_strategy: str = "average",
    parallel: bool = True,
    **kwargs  # Same as Lowess
)
```

**Methods:**

#### `process_chunk`

Processes a chunk of data.

```python
def process_chunk(x: ArrayLike, y: ArrayLike) -> LowessResult
```

#### `finalize`

Finalizes the smoothing process and returns any remaining buffered data.

```python
def finalize() -> LowessResult
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
streaming = fl.StreamingLowess(chunk_size=10000, overlap=1000)
chunk_result = streaming.process_chunk(x_chunk, y_chunk)
final_result = streaming.finalize()
```

---

### OnlineLowess

Stateful online mode for real-time data.

```python
class fastlowess.OnlineLowess(
    fraction: float = 0.2,
    window_capacity: int = 100,
    min_points: int = 2,
    iterations: int = 3,
    update_mode: str = "incremental",
    **kwargs  # Same as Lowess
)
```

**Methods:**

#### `add_points`

Adds new points to the window and returns smoothed values.

```python
def add_points(x: ArrayLike, y: ArrayLike) -> LowessResult
```

**Additional Parameters:**

| Parameter         | Type | Default       | Description          |
|-------------------|------|---------------|----------------------|
| `window_capacity` | int  | 100           | Max points in window |
| `min_points`      | int  | 2             | Points before output |
| `update_mode`     | str  | "incremental" | Update strategy      |

**Example:**

```python
online = fl.OnlineLowess(window_capacity=25)
result = online.add_points(new_x, new_y)
print(result.y)
```

---

## Return Object

### LowessResult

The result object returned by `fit()` and other methods.

| Attribute            | Type    | Description                         |
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
| `diagnostics`        | object  | If `return_diagnostics=True`        |
| `cv_scores`          | list    | If cross-validation used            |

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

When `return_diagnostics=True`, the `result.diagnostics` attribute contains:

```python
class Diagnostics:
    rmse: float        # Root Mean Square Error
    mae: float         # Mean Absolute Error
    r_squared: float   # R² coefficient
    residual_sd: float # Residual standard deviation
    effective_df: float # Effective degrees of freedom
```
