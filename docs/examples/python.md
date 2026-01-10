# Python Examples

Complete Python examples demonstrating fastlowess capabilities with NumPy and matplotlib.

## Batch Smoothing

Process complete datasets with confidence intervals, diagnostics, and cross-validation.

```python
--8<-- "examples/python/batch_smoothing.py"
```

[:material-download: Download batch_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/examples/python/batch_smoothing.py)

---

## Streaming Smoothing

Process large datasets in memory-efficient chunks with overlap merging.

```python
--8<-- "examples/python/streaming_smoothing.py"
```

[:material-download: Download streaming_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/examples/python/streaming_smoothing.py)

---

## Online Smoothing

Real-time smoothing with sliding window for streaming data applications.

```python
--8<-- "examples/python/online_smoothing.py"
```

[:material-download: Download online_smoothing.py](https://github.com/thisisamirv/lowess-project/blob/main/examples/python/online_smoothing.py)

---

## Running the Examples

```bash
# Install dependencies
pip install fastlowess matplotlib numpy

# Run examples
cd examples/python
python batch_smoothing.py
python streaming_smoothing.py
python online_smoothing.py
```

## Output

The batch smoothing example generates visualization plots in `examples/python/plots/`:

- `batch_main.png` - Main smoothing comparison
- `batch_weights.png` - Robustness weights visualization
- `batch_boundary.png` - Boundary policy comparison
