# Python Examples

Complete Python examples demonstrating fastlowess capabilities with NumPy and matplotlib.

## Batch Smoothing

Process complete datasets with confidence intervals, diagnostics, and cross-validation.

```python
--8<-- "bindings/python/examples/batch_smoothing.py"
```

[:material-download: Download batch_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/bindings/python/examples/batch_smoothing.py)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks with overlap merging.

```python
--8<-- "bindings/python/examples/streaming_smoothing.py"
```

[:material-download: Download streaming_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/bindings/python/examples/streaming_smoothing.py)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data applications.

```python
--8<-- "bindings/python/examples/online_smoothing.py"
```

[:material-download: Download online_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/bindings/python/examples/online_smoothing.py)

---

## Running the Examples

```bash
# Install dependencies
pip install fastlowess matplotlib numpy

# Run examples
cd bindings/python/examples
python batch_smoothing.py
python streaming_smoothing.py
python online_smoothing.py
```

## Output

The batch smoothing example generates visualization plots in `bindings/python/examples/plots/`:

- `batch_main.png` - Main smoothing comparison
- `batch_weights.png` - Robustness weights visualization
- `batch_boundary.png` - Boundary policy comparison
