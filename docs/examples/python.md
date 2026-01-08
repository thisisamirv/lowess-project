# Python Examples

Complete code examples for the `fastlowess` Python package.

## Basic Smoothing

```python
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

# Generate noisy data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.3, 100)

# Smooth
result = fl.smooth(x, y, fraction=0.3, iterations=3)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, result["y"], "r-", lwd=2, label="LOWESS")
plt.legend()
plt.show()
```

---

## With Confidence Intervals

```python
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.3, 100)

result = fl.smooth(
    x, y,
    fraction=0.3,
    iterations=3,
    confidence_intervals=0.95,
    prediction_intervals=0.95,
    return_diagnostics=True
)

# Plot with bands
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.3, label="Data")
plt.plot(x, result["y"], "b-", lwd=2, label="Smoothed")

plt.fill_between(
    x, result["confidence_lower"], result["confidence_upper"],
    alpha=0.3, color="blue", label="95% CI"
)
plt.fill_between(
    x, result["prediction_lower"], result["prediction_upper"],
    alpha=0.1, color="blue", label="95% PI"
)

plt.legend()
plt.title(f"R² = {result['diagnostics']['r_squared']:.4f}")
plt.show()
```

---

## Cross-Validation

```python
import fastlowess as fl
import numpy as np

x = np.linspace(0, 20, 200)
y = x * 0.5 + np.sin(x) + np.random.normal(0, 1, 200)

result = fl.smooth(
    x, y,
    cv_method="kfold",
    cv_k=5,
    cv_fractions=[0.1, 0.2, 0.3, 0.5, 0.7],
    cv_seed=42
)

print(f"Best fraction: {result['fraction_used']}")
print(f"CV scores: {result['cv_scores']}")
```

---

## Outlier Detection

```python
import fastlowess as fl
import numpy as np

x = np.arange(50, dtype=float)
y = x * 2.0
y[10] = 100  # Outlier
y[25] = -50  # Outlier
y[40] = 150  # Outlier

result = fl.smooth(
    x, y,
    fraction=0.3,
    iterations=5,
    robustness_method="bisquare",
    return_robustness_weights=True
)

# Find outliers (low weight)
outliers = [i for i, w in enumerate(result["robustness_weights"]) if w < 0.5]
print(f"Outliers at indices: {outliers}")
```

---

## Streaming Large Data

```python
import fastlowess as fl
import numpy as np

# Large dataset
n = 1_000_000
x = np.arange(n, dtype=float)
y = np.sin(x / 10000) + np.random.normal(0, 0.1, n)

# Streaming mode handles chunking internally
result = fl.smooth_streaming(
    x, y,
    fraction=0.01,
    chunk_size=50000,
    overlap=5000,
    merge_strategy="weighted",
    parallel=True
)

print(f"Processed {len(result['y'])} points")
```

---

## Online (Real-Time) Processing

```python
import fastlowess as fl
import numpy as np

# Sensor simulation
times = np.arange(100, dtype=float)
values = 20 + 5 * np.sin(times / 10) + np.random.normal(0, 1, 100)

result = fl.smooth_online(
    times, values,
    fraction=0.3,
    window_capacity=25,
    min_points=5,
    update_mode="incremental"
)

# Each output corresponds to a time point after warm-up
print(f"Output values: {len(result['y'])}")
```

---

## Multiple Fractions Comparison

```python
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.3, 100)

fractions = [0.1, 0.3, 0.5, 0.7]

plt.figure(figsize=(12, 5))
plt.scatter(x, y, alpha=0.3, s=10, label="Data")

for f in fractions:
    result = fl.smooth(x, y, fraction=f)
    plt.plot(x, result["y"], label=f"f={f}")

plt.legend()
plt.title("Effect of Fraction Parameter")
plt.show()
```

---

## Kernel Comparison

```python
import fastlowess as fl
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)

kernels = ["tricube", "epanechnikov", "gaussian", "uniform"]

plt.figure(figsize=(12, 5))
plt.scatter(x, y, alpha=0.3, s=10)

for kernel in kernels:
    result = fl.smooth(x, y, fraction=0.3, weight_function=kernel)
    plt.plot(x, result["y"], label=kernel)

plt.legend()
plt.title("Kernel Function Comparison")
plt.show()
```

---

## Pandas Integration

```python
import fastlowess as fl
import pandas as pd
import numpy as np

# Create DataFrame
df = pd.DataFrame({
    "time": np.arange(100),
    "value": np.sin(np.arange(100) / 10) + np.random.normal(0, 0.2, 100)
})

# Smooth
result = fl.smooth(df["time"].values, df["value"].values, fraction=0.3)

# Add to DataFrame
df["smoothed"] = result["y"]

print(df.head())
```

---

## Error Handling

```python
import fastlowess as fl
import numpy as np

try:
    # Invalid fraction
    result = fl.smooth([1, 2, 3], [1, 2, 3], fraction=2.0)
except ValueError as e:
    print(f"Error: {e}")

try:
    # Mismatched lengths
    result = fl.smooth([1, 2, 3], [1, 2])
except ValueError as e:
    print(f"Error: {e}")
```
